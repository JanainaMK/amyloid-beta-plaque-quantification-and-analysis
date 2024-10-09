from src.util.warnings_filter import suppress_plbolt_warnings

suppress_plbolt_warnings()  # included to avoid logging spam. exclude to view all warnings
import os

import numpy as np
import torch

from src.categorisation.supervised.linear_classifier import LinearEvaluation


class EnsembleModel:
    def __init__(
        self,
        train_size: int,
        n_folds: int,
        start_version: int,
        save_dir: str,
        name: str,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs,
    ):
        self.device = device
        self.model_name = name

        eval_path_sub = f"evaluation/{self.model_name}/train_size_{train_size}"
        self.fold_thresholds = torch.tensor(
            np.load(f"{eval_path_sub}/{start_version}_fold_thresholds.npy")
        )
        print("Loaded fold thresholds")
        print(self.fold_thresholds)
        self.fold_thresholds = self.fold_thresholds.to(self.device)

        fold_versions = [
            "version_" + str(fold_version)
            for fold_version in range(start_version, n_folds + start_version)
        ]
        print(f"model folds used for train_size {train_size}:", fold_versions)

        # Initialize pre-trained models for ensemble
        self.models = []
        with torch.no_grad():
            for fold in fold_versions:
                model_path = save_dir + self.model_name + "/" + fold + "/checkpoints/"
                first_epoch_file = next(
                    (
                        file
                        for file in os.listdir(model_path)
                        if file.startswith("epoch")
                    ),
                    None,
                )
                model = LinearEvaluation.load_from_checkpoint(
                    model_path + first_epoch_file
                )
                model.eval()
                model.to(self.device)
                self.models.append(model)

    def __call__(self, batch):
        # Collect predictions from all models
        model_predictions = []
        with torch.no_grad():
            for model_idx, model in enumerate(self.models):
                probabilities = model(batch)

                # Convert probabilities to binary values with True indicating that the threshold requirement is met, False otherwise
                satisfied_threshold_mask = (
                    probabilities >= self.fold_thresholds[model_idx]
                )
                # Keep track which samples have at least 1 class satisfying the threshold
                satisfied_sample_condition = (
                    satisfied_threshold_mask.int().sum(dim=1) > 0
                )
                # Select max probability class in case no class satisfies the threshold
                notsat_max_prob_predictions_thresh = torch.argmax(probabilities, dim=1)
                # Select max probability class that satisfies the threshold
                probabilities[~satisfied_threshold_mask] = -1
                sat_max_prob_predictions_thresh = torch.argmax(probabilities, dim=1)

                assigned_classes = torch.where(
                    satisfied_sample_condition,
                    sat_max_prob_predictions_thresh,
                    notsat_max_prob_predictions_thresh,
                )
                model_predictions.append(assigned_classes)

        # Stack predictions, shape: (num_models, batch_size)
        predictions = torch.stack(model_predictions)
        # Apply majority vote, shape: (batch_size)
        majority_vote, _ = torch.mode(predictions, dim=0)

        return majority_vote
