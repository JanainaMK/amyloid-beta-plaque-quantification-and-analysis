from src.util.warnings_filter import suppress_plbolt_warnings

suppress_plbolt_warnings()  # included to avoid logging spam. exclude to view all wanings
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torchsummary import summary

from src.categorisation.autoencoders.vae import VAE
from src.categorisation.plaques_dataset import PlaquesDataset
from src.util.cli import parse_config


def main():
    config = parse_config()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datamodule = PlaquesDataset(
        **config["data_params"], pin_memory=len(config["run_params"]["gpus"]) != 0
    )
    datamodule.setup()

    patch_size = config["data_params"]["patch_size"]
    model = VAE(input_height=patch_size)

    print(VAE.pretrained_weights_available())
    model = model.from_pretrained("cifar10-resnet18")
    model.to(device)

    summary(model, input_size=(3, patch_size, patch_size))

    tb_logger = TensorBoardLogger(
        save_dir=config["logging_params"]["save_dir"],
        name=config["logging_params"]["name"],
    )
    monitor_loss_type = "val_loss"
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        save_last=True, save_top_k=1, monitor=monitor_loss_type
    )
    early_stop = EarlyStopping(
        monitor=monitor_loss_type,  # Set monitoring for validation loss
        min_delta=1e-9,  # Minimum change in validation loss
        patience=5,  # Number of epochs
        verbose=True,  # Print logs
        mode="min",  # Minimize loss
    )
    callbacks = [model_checkpoint, lr_monitor, early_stop]

    trainer = Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        strategy=DDPStrategy(
            process_group_backend="gloo", find_unused_parameters=False
        ),
        **config["trainer_params"],
    )

    Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/reconstructions").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/originals").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['logging_params']['name']} =======")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
