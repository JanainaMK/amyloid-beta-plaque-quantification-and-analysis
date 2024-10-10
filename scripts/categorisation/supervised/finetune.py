from src.util.warnings_filter import suppress_plbolt_warnings

suppress_plbolt_warnings()  # included to avoid logging spam. exclude to view all wanings
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from src.categorisation.cli import parse_config
from src.categorisation.supervised.labeled_plaques_dataset import LabeledPlaquesDataset
from src.categorisation.supervised.linear_classifier import LinearEvaluation


def main():
    config = parse_config()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_folds = config["data_params"]["n_folds"]
    train_sizes = config["data_params"]["train_sizes"]
    batch_size = config["data_params"]["train_batch_size"]
    num_classes = config["model_params"]["num_classes"]

    for train_size in train_sizes:
        print(f"train_size {train_size}")
        datamodule = LabeledPlaquesDataset(
            **config["data_params"],
            pin_memory=len(config["run_params"]["gpus"]) != 0,
        )
        num_samples = train_size * n_folds * num_classes
        for fold_idx in range(n_folds):
            print(f"fold_idx {fold_idx+1}/{n_folds}")
            linear_model = LinearEvaluation(
                num_samples=num_samples,
                batch_size=batch_size,
                **config["model_params"],
            )

            model = linear_model.to(device)

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
                min_delta=1e-7,  # Minimum change in validation loss
                patience=5,  # Number of epochs
                verbose=True,  # Print logs
                mode="min",  # Minimize loss
            )
            callbacks = [model_checkpoint, lr_monitor, early_stop]

            trainer = Trainer(
                logger=tb_logger,
                callbacks=callbacks,
                log_every_n_steps=int((num_samples / batch_size) / 2),
                strategy=DDPStrategy(process_group_backend="gloo"),
                **config["trainer_params"],
            )

            trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
