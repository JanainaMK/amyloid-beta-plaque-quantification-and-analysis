import os
import urllib.parse
from argparse import ArgumentParser

import torch
import torchvision.utils as vutils
from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F  # noqa: N812


# based on lightning-bolts VAE and https://github.com/AntixK/PyTorch-VAE architecture
class VAE(LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.

    Model is available pretrained on different datasets:

    Example::

        # not pretrained
        vae = VAE()

        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')

        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        "cifar10-resnet18": urllib.parse.urljoin(
            _HTTPS_AWS_HUB, "vae/vae-cifar10/checkpoints/epoch%3D89.ckpt"
        ),
        "stl10-resnet18": urllib.parse.urljoin(
            _HTTPS_AWS_HUB, "vae/vae-stl10/checkpoints/epoch%3D89.ckpt"
        ),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ) -> None:
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            "resnet18": {
                "enc": resnet18_encoder,
                "dec": resnet18_decoder,
            },
            "resnet50": {
                "enc": resnet50_encoder,
                "dec": resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(
                self.latent_dim, self.input_height, first_conv, maxpool1
            )
        else:
            self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]["dec"](
                self.latent_dim, self.input_height, first_conv, maxpool1
            )

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + " not present in pretrained weights.")

        return self.load_from_checkpoint(
            VAE.pretrained_urls[checkpoint_name], strict=False
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def generate_samples(self, num_samples: int, current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decoder(z)
        return samples

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        img, _ = batch
        self.curr_device = img.device
        loss, logs = self.step(batch, batch_idx)
        # logs for tensorboard
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        for key, val in logs.items():
            self.logger.experiment.add_scalars(
                f"{key}", {"train": val.item()}, global_step=self.global_step
            )
        return loss

    def validation_step(self, batch, batch_idx):
        img, _ = batch
        self.curr_device = img.device
        loss, logs = self.step(batch, batch_idx)
        # logs for tensorboard
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        for key, val in logs.items():
            self.logger.experiment.add_scalars(
                f"{key}", {"val": val.item()}, global_step=self.global_step
            )
        return loss

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)

        self.eval()
        with torch.no_grad():
            # save original images without modifications
            vutils.save_image(
                test_input,
                os.path.join(
                    self.logger.log_dir,
                    "originals",
                    f"orig_{self.logger.name}_Epoch_{self.current_epoch}.png",
                ),
                normalize=True,
                nrow=12,
            )
            # save reconstruction of original images
            recons = self.forward(test_input)
            vutils.save_image(
                recons.cpu().data,
                os.path.join(
                    self.logger.log_dir,
                    "reconstructions",
                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
                ),
                normalize=True,
                nrow=12,
            )

            # save samples generated based on vae
            try:
                samples = self.generate_samples(144, self.curr_device)
                vutils.save_image(
                    samples.cpu().data,
                    os.path.join(
                        self.logger.log_dir,
                        "samples",
                        f"{self.logger.name}_Epoch_{self.current_epoch}.png",
                    ),
                    normalize=True,
                    nrow=12,
                )
            except Warning:
                pass
        self.train()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--enc_type", type=str, default="resnet18", help="resnet18/resnet50"
        )
        parser.add_argument("--first_conv", action="store_true")
        parser.add_argument("--maxpool1", action="store_true")
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets",
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser
