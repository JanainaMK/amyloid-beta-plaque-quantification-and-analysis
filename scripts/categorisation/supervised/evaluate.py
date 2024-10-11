from src.util.warnings_filter import suppress_plbolt_warnings

suppress_plbolt_warnings()  # included to avoid logging spam. exclude to view all wanings
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.categorisation.cli import parse_config
from src.categorisation.supervised.labeled_plaques_dataset import LabeledPlaquesDataset
from src.categorisation.supervised.linear_classifier import LinearEvaluation
from src.categorisation.supervised.plaque_labels import get_label_names


def main():
    config = parse_config()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = config["model_params"]["num_classes"]
    n_folds = config["data_params"]["n_folds"]
    train_sizes = config["data_params"]["train_sizes"]
    start_version = config["evaluate_params"]["start_version"]
    save_dir = config["logging_params"]["save_dir"]
    model_name = config["logging_params"]["name"]
    generate_thresholds = config["evaluate_params"]["generate_thresholds"]
    primary_labels = config["evaluate_params"]["primary_labels"]
    anomaly_labels = config["evaluate_params"]["anomaly_labels"]
    class_names = get_label_names()

    class_labels = list(range(num_classes))

    # models trained on the split folds.
    # train instance consists of the fold models trained with the specified size.
    # start version indicates which saved model version to start from.
    model_trainsize_folds = [
        [
            "version_" + str(fold_version + train_size_instance * n_folds)
            for fold_version in range(start_version, n_folds + start_version)
        ]
        for train_size_instance in range(0, len(train_sizes))
    ]

    overall_auc = []
    overall_acc = []
    overall_rec = []
    overall_prec = []
    overall_f1 = []

    primary_auc = []
    primary_acc = []
    primary_rec = []
    primary_prec = []
    primary_f1 = []

    anomaly_auc = []
    anomaly_acc = []
    anomaly_rec = []
    anomaly_prec = []
    anomaly_f1 = []

    class_rec = np.zeros((num_classes, n_folds))
    class_prec = np.zeros((num_classes, n_folds))
    class_f1 = np.zeros((num_classes, n_folds))
    class_roc_auc = np.zeros((num_classes, n_folds))

    with torch.no_grad():
        for trainsize_idx, trainsize_folds in enumerate(model_trainsize_folds):
            train_size = train_sizes[trainsize_idx]
            print(f"model folds used for train_size {train_size}:", trainsize_folds)
            datamodule = LabeledPlaquesDataset(
                **config["data_params"],
                pin_memory=len(config["run_params"]["gpus"]) != 0,
            )
            eval_path_sub = f"evaluation/{model_name}/train_size_{train_size}"

            if generate_thresholds:
                fold_thresholds = np.zeros((n_folds, num_classes))
            else:
                print("Loaded fold thresholds")
                fold_thresholds = np.load(
                    f"{eval_path_sub}/{start_version}_fold_thresholds.npy"
                )
                print(fold_thresholds)

            total_cm = None

            for fold_idx, fold in enumerate(trainsize_folds):
                print(f"fold {fold_idx+1}/{len(trainsize_folds)}, {fold}")
                datamodule.setup()

                model_path = save_dir + model_name + "/" + fold + "/checkpoints/"
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
                model.to(device)
                model.eval()

                if generate_thresholds:
                    # Use validation set to generate thresholds per fold
                    eval_mode = "val"
                    dataloader = datamodule.val_dataloader()
                else:
                    eval_mode = "test"
                    dataloader = datamodule.test_dataloader()

                print(f"Evaluating on {eval_mode} set")

                fold_labels = []
                fold_probabilities = []
                for _, batch in enumerate(dataloader):
                    imgs, labels, _ = batch
                    imgs = imgs.to(device)
                    probabilities = model(imgs)

                    fold_labels.extend(labels)
                    fold_probabilities.extend(probabilities.cpu().detach())

                if generate_thresholds:
                    y_true_bin = label_binarize(fold_labels, classes=class_labels)
                    # Compute ROC curve for each class
                    thresholds = dict()
                    for i in range(num_classes):
                        fpr, tpr, thresholds[i] = roc_curve(
                            y_true_bin[:, i], np.array(fold_probabilities)[:, i]
                        )
                        # Find optimal index based on minimum distance
                        distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
                        optimal_idx = np.argmin(distances)
                        fold_thresholds[fold_idx][i] = thresholds[i][optimal_idx]

                fold_thresholds = fold_thresholds.round(decimals=4)
                opt_class_thresholds = fold_thresholds[fold_idx]
                print(f"thresholds used for fold {fold}:\n {opt_class_thresholds}")

                fold_assignments = []
                for sample_prob in fold_probabilities:
                    max_prob_class = None
                    max_probability = -1

                    # For a sample assign the class with highest probability that is at least higher than its threshold
                    for class_index in range(num_classes):
                        if (
                            sample_prob[class_index]
                            >= opt_class_thresholds[class_index]
                        ):
                            if sample_prob[class_index] > max_probability:
                                max_probability = sample_prob[class_index]
                                max_prob_class = class_index
                    # Assign class index with max probability that satisfies threshold for the current sample
                    # If threshold requirement is not met for any class, select argmax of sample probabilities
                    fold_assignments.append(
                        max_prob_class
                        if max_prob_class is not None
                        else np.argmax(sample_prob)
                    )

                conf_matrix = confusion_matrix(fold_labels, fold_assignments)
                if total_cm is None:
                    total_cm = conf_matrix
                else:
                    total_cm += conf_matrix

                fold_acc = accuracy_score(fold_labels, fold_assignments)
                print("fold accuracy:", fold_acc)

                y_true_bin = label_binarize(fold_labels, classes=class_labels)

                # Collect overall metrics
                overall_acc.append(fold_acc)
                overall_rec.append(
                    recall_score(fold_labels, fold_assignments, average="weighted")
                )
                overall_prec.append(
                    precision_score(fold_labels, fold_assignments, average="weighted")
                )
                overall_f1.append(
                    f1_score(fold_labels, fold_assignments, average="weighted")
                )
                overall_auc.append(
                    roc_auc_score(
                        y_true_bin,
                        fold_probabilities,
                        average="weighted",
                        multi_class="ovr",
                    )
                )

                # Collect primary metrics
                rs = recall_score(
                    fold_labels,
                    fold_assignments,
                    average="weighted",
                    labels=primary_labels,
                )
                primary_acc.append(rs)  # weighted recall equals accuracy
                primary_rec.append(rs)
                primary_prec.append(
                    precision_score(
                        fold_labels,
                        fold_assignments,
                        average="weighted",
                        labels=primary_labels,
                    )
                )
                primary_f1.append(
                    f1_score(
                        fold_labels,
                        fold_assignments,
                        average="weighted",
                        labels=primary_labels,
                    )
                )

                # Collect anomaly metrics
                rs = recall_score(
                    fold_labels,
                    fold_assignments,
                    average="weighted",
                    labels=anomaly_labels,
                )
                anomaly_acc.append(rs)  # weighted recall equals accuracy
                anomaly_rec.append(rs)
                anomaly_prec.append(
                    precision_score(
                        fold_labels,
                        fold_assignments,
                        average="weighted",
                        labels=anomaly_labels,
                    )
                )
                anomaly_f1.append(
                    f1_score(
                        fold_labels,
                        fold_assignments,
                        average="weighted",
                        labels=anomaly_labels,
                    )
                )

                # Compute per class metrics
                for class_idx in range(num_classes):
                    cp = precision_score(
                        fold_labels, fold_assignments, labels=[class_idx], average=None
                    )
                    crec = recall_score(
                        fold_labels, fold_assignments, labels=[class_idx], average=None
                    )
                    cf1 = f1_score(
                        fold_labels, fold_assignments, labels=[class_idx], average=None
                    )

                    class_prec[class_idx, fold_idx] = cp[0]
                    class_rec[class_idx, fold_idx] = crec[0]
                    class_f1[class_idx, fold_idx] = cf1[0]

                class_roc_auc[:, fold_idx] = roc_auc_score(
                    y_true_bin, fold_probabilities, average=None, multi_class="ovr"
                )

                primary_auc.append(np.mean(class_roc_auc[primary_labels, fold_idx]))
                anomaly_auc.append(np.mean(class_roc_auc[anomaly_labels, fold_idx]))

            os.makedirs(eval_path_sub, exist_ok=True)
            if generate_thresholds:
                print("Generated and saved fold thresholds")
                print(fold_thresholds)
                np.save(
                    f"{eval_path_sub}/{start_version}_fold_thresholds.npy",
                    fold_thresholds,
                )

            # Log metrics
            eval_path = f"{eval_path_sub}/{eval_mode}"
            os.makedirs(eval_path, exist_ok=True)
            log_file_path = f"{eval_path}/{start_version}_metrics.txt"
            with open(log_file_path, "w") as log_file:

                def log_print(message):
                    print(message)
                    log_file.write(message + "\n")

                log_print("Overall Metrics:")
                log_print(
                    f"  accuracy: mean {np.mean(overall_acc) * 100:.2f}, std {np.std(overall_acc, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  recall: mean {np.mean(overall_rec) * 100:.2f}, std {np.std(overall_rec, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  precision: mean {np.mean(overall_prec) * 100:.2f}, std {np.std(overall_prec, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  f1: mean {np.mean(overall_f1) * 100:.2f}, std {np.std(overall_f1, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  roc auc: mean {np.mean(overall_auc) * 100:.2f}, std {np.std(overall_auc, ddof=1) * 100:.2f}"
                )

                log_print("Primary Metrics:")
                log_print(
                    f"  accuracy: mean {np.mean(primary_acc) * 100:.2f}, std {np.std(primary_acc, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  recall: mean {np.mean(primary_rec) * 100:.2f}, std {np.std(primary_rec, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  precision: mean {np.mean(primary_prec) * 100:.2f}, std {np.std(primary_prec, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  f1: mean {np.mean(primary_f1) * 100:.2f}, std {np.std(primary_f1, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  roc auc: mean {np.mean(primary_auc) * 100:.2f}, std {np.std(primary_auc, ddof=1) * 100:.2f}"
                )

                log_print("Anomaly Metrics:")
                log_print(
                    f"  accuracy: mean {np.mean(anomaly_acc) * 100:.2f}, std {np.std(anomaly_acc, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  recall: mean {np.mean(anomaly_rec) * 100:.2f}, std {np.std(anomaly_rec, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  precision: mean {np.mean(anomaly_prec) * 100:.2f}, std {np.std(anomaly_prec, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  f1: mean {np.mean(anomaly_f1) * 100:.2f}, std {np.std(anomaly_f1, ddof=1) * 100:.2f}"
                )
                log_print(
                    f"  roc auc: mean {np.mean(anomaly_auc) * 100:.2f}, std {np.std(anomaly_auc, ddof=1) * 100:.2f}"
                )

                log_print("\n")

                avg_precisions = np.mean(class_prec, axis=1) * 100
                std_precisions = np.std(class_prec, axis=1, ddof=1) * 100

                avg_recalls = np.mean(class_rec, axis=1) * 100
                std_recalls = np.std(class_rec, axis=1, ddof=1) * 100

                avg_f1s = np.mean(class_f1, axis=1) * 100
                std_f1s = np.std(class_f1, axis=1, ddof=1) * 100

                avg_roc_aucs = np.mean(class_roc_auc, axis=1) * 100
                std_roc_aucs = np.std(class_roc_auc, axis=1, ddof=1) * 100

                for i, class_name in enumerate(class_names):
                    log_print(f"{class_name}:")
                    log_print(
                        f"  recall: mean {avg_recalls[i]:.2f}, std {std_recalls[i]:.2f}"
                    )
                    log_print(
                        f"  precision: mean {avg_precisions[i]:.2f}, std {std_precisions[i]:.2f}"
                    )
                    log_print(f"  f1: mean {avg_f1s[i]:.2f}, std {std_f1s[i]:.2f}")
                    log_print(
                        f"  roc auc: mean {avg_roc_aucs[i]:.2f}, std {std_roc_aucs[i]:.2f}"
                    )

            # Confusion matrix
            # Normalize confusion matrix to percentages
            cm_normalized = (
                total_cm.astype("float") / total_cm.sum(axis=1)[:, np.newaxis] * 100
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm_normalized, display_labels=class_names
            )
            _, ax = plt.subplots(figsize=(8, 6))
            disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=ax)
            ax.set_xlabel("Predicted label (%)")
            ax.set_ylabel("True label (%)")
            ax.set_title("Confusion Matrix")
            for i in range(cm_normalized.shape[0]):
                for j in range(cm_normalized.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm_normalized[i, j], ".2f"),
                        ha="center",
                        va="center",
                        color="white" if cm_normalized[i, j] > 50 else "black",
                    )

            plt.title("Precision vs Recall for Each Class")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tight_layout()

            plt.savefig(f"{eval_path}/{start_version}_confusion_matrix.png", dpi=1200)


if __name__ == "__main__":
    main()
