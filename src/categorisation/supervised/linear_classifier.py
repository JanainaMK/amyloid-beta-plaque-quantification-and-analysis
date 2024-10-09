import os

import requests
import torch
import torchvision.models as torchvision_models
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import softmax


def download_model(url, model_path):
    print("Downloading model")
    response = requests.get(url)
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            f.write(response.content)
        print(f"Model saved to {model_path}")
    else:
        print(f"Model download failed: {response.status_code}")


class LinearEvaluation(LightningModule):
    def __init__(
        self,
        num_classes,
        model_type: str,
        num_samples: int,
        batch_size: int,
        arch: str = "resnet50",
        num_nodes: int = 1,
        hidden_dim: int = 2048,
        gpus: int = 1,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        dropout_probability: float = 0.2,
        **kwargs,
    ) -> None:
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.num_nodes = num_nodes
        self.num_samples = num_samples

        self.num_classes = num_classes
        self.model_type = model_type
        self.dropout_probability = dropout_probability
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.arch = arch

        # compute iters per epoch
        global_batch_size = (
            self.num_nodes * self.gpus * self.batch_size
            if self.gpus > 0
            else self.batch_size
        )
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(self.hidden_dim, self.num_classes, bias=True),
        )

        if self.model_type == "simclr":
            weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
            model = SimCLR.load_from_checkpoint(weight_path, strict=False)
            self.encoder = model.encoder
        elif self.model_type == "moco":

            model = torchvision_models.__dict__[self.arch]()
            linear_keyword = "fc"

            getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
            getattr(model, linear_keyword).bias.data.zero_()

            pretrained_file = "models/r-50-1000ep.pth.tar"
            if not os.path.exists(pretrained_file):
                # source: https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md
                url = "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar"
                download_model(url, pretrained_file)
            if os.path.isfile(pretrained_file):
                checkpoint = torch.load(pretrained_file, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint["state_dict"]
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith("module.base_encoder") and not k.startswith(
                        "module.base_encoder.%s" % linear_keyword
                    ):
                        # remove prefix
                        state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                model.load_state_dict(state_dict, strict=False)

                self.encoder = nn.Sequential(*list(model.children())[:-1])
            else:
                raise Exception(f"{pretrained_file} is not a recognized file path.")
        else:
            raise Exception(f"{self.model_type} is not a recognized model.")

        for param in self.encoder.parameters():
            param.requires_grad = True

        print(f"loaded pre-trained model {self.model_type}")

    def logits(self, x):
        if self.model_type == "simclr":
            encoding = self.encoder(x)[-1]
        elif self.model_type == "moco":
            encoding = self.encoder(x)

        # Forward pass through classification head
        logits = self.classification_head(encoding)
        return logits

    def forward(self, x):
        logits = self.logits(x)
        probabilities = softmax(logits, dim=1)
        return probabilities

    def shared_step(self, batch):
        inputs, labels, _ = batch
        logits = self.logits(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = self.parameters()

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
