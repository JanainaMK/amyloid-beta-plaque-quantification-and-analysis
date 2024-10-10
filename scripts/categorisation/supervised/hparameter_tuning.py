from src.util.warnings_filter import suppress_plbolt_warnings

suppress_plbolt_warnings()  # included to avoid logging spam. exclude to view all wanings
import numpy as np
import optuna
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from src.categorisation.cli import parse_config
from src.categorisation.supervised.labeled_plaques_dataset import LabeledPlaquesDataset
from src.categorisation.supervised.linear_classifier import LinearEvaluation


class ValidationLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.validation_losses = []
        self.best_val_loss = 1000

    def on_validation_end(self, trainer, pl_module, **kwargs):
        # Save best validation loss so far
        curr_val_loss = trainer.callback_metrics["val_loss"]
        if curr_val_loss < self.best_val_loss:
            self.best_val_loss = curr_val_loss

    def get_val_loss(self):
        return self.best_val_loss.item()


def objective(trial, config, device):
    # Hyperparameters to tune
    learning_rate = trial.suggest_categorical("weight_decay", [1e-5, 1e-4])
    weight_decay = trial.suggest_categorical("learning_rate", [1e-5, 1e-4])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    dropout_probability = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])

    tuned_params = [
        "learning_rate",
        "weight_decay",
        "batch_size",
        "dropout_probability",
    ]
    filtered_data_params = {
        key: value
        for key, value in config["model_params"].items()
        if key not in tuned_params
    }
    num_classes = config["model_params"]["num_classes"]

    n_folds = config["data_params"]["n_folds"]
    train_sizes = config["data_params"]["train_sizes"]

    val_losses = []
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
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                dropout_probability=dropout_probability,
                **filtered_data_params,
            )

            model = linear_model.to(device)

            monitor_loss_type = "val_loss"
            early_stop = EarlyStopping(
                monitor=monitor_loss_type,  # Set monitoring for validation loss
                min_delta=1e-7,  # Minimum change in validation loss
                patience=5,  # Number of epochs
                verbose=True,  # Print logs
                mode="min",  # Minimize loss
            )
            validation_loss_callback = ValidationLossCallback()
            callbacks = [early_stop, validation_loss_callback]

            trainer = Trainer(
                callbacks=callbacks,
                log_every_n_steps=int((num_samples / batch_size) / 2),
                strategy=DDPStrategy(
                    process_group_backend="gloo", find_unused_parameters=False
                ),
                **config["trainer_params"],
            )

            trainer.fit(model, datamodule=datamodule)
            best_curr_val_loss = validation_loss_callback.get_val_loss()
            val_losses.append(best_curr_val_loss)

    mean_val_loss = np.mean(val_losses)
    return mean_val_loss


def print_best_trial_callback(study, trial):
    print(
        f"Best trial so far: \n Value: {study.best_trial.value}, Params: {study.best_trial.params}"
    )


def main():
    config = parse_config()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pruner = optuna.pruners.MedianPruner() if True else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, config=config, device=device),
        n_trials=config["data_params"]["n_trials"],
        callbacks=[print_best_trial_callback],
    )

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
