import os
import shutil
from math import ceil

import numpy as np
import torch

from src.categorisation.plaques_dataset import PlaquesDataset
from src.categorisation.supervised.ensemble_model import EnsembleModel
from src.categorisation.supervised.labeled_plaques_dataset import LabeledPlaquesDataset
from src.categorisation.cli import parse_config


def main():
    config = parse_config()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # highest interpretable diameter size in microns for a plaque
    max_size_micron = 200
    max_size = int(max_size_micron / 0.274)
    max_area = (max_size / 2) ** 2 * np.pi
    print(f"maximum permitted plaque area: {max_area}")

    num_classes = config["model_params"]["num_classes"]
    batch_size = config["data_params"]["train_batch_size"]
    log_name = config["logging_params"]["name"]
    assignment_save_path = config["prediction_params"]["assignment_save_path"]
    use_labeled_data = config["prediction_params"]["use_labeled_data"]
    temp_dir = "result/temp"
    temp_save_path = f"{temp_dir}/{log_name}"

    if use_labeled_data:
        datamodule = LabeledPlaquesDataset(
            **config["data_params"],
            pin_memory=len(config["run_params"]["gpus"]) != 0,
        )
        datamodule.setup()
    else:
        datamodule = PlaquesDataset(
            **config["data_params"], pin_memory=len(config["run_params"]["gpus"]) != 0
        )
        datamodule.setup()

    # load ensemble model
    model = EnsembleModel(
        train_size=config["data_params"]["train_sizes"][0],
        device=device,
        **config["data_params"],
        **config["evaluate_params"],
        **config["logging_params"],
    )

    if use_labeled_data:
        log_name += "_true"

    all_assignments = []
    all_file_names = []
    all_assignments = []
    all_local_indices = []

    dataloader = datamodule.all_dataloader()
    len_dataloader = len(dataloader)
    stop = len_dataloader + 1

    samples_div = 400000
    batch_div = max(int(len_dataloader * batch_size / samples_div), 1)
    batch_div_amount = ceil(len_dataloader / batch_div)
    print("batch_div", batch_div)
    print("batch_div_amount", batch_div_amount)

    os.makedirs(assignment_save_path, exist_ok=True)
    os.makedirs(temp_save_path, exist_ok=True)

    save_indices = []
    for i, batch in enumerate(dataloader):
        if i >= stop:
            break

        if i == 0:
            print(f"batch {i+1}/{len_dataloader}")

        if use_labeled_data:
            imgs, _, batch_data = batch
        else:
            imgs, batch_data = batch
        imgs = imgs.to(device)
        assignments = model(imgs)

        file_names = batch_data[0]
        local_indices = batch_data[1]
        areas = batch_data[2]
        # assign plaques that are too big to the undefined unknown class
        assignments[areas > max_area] = num_classes - 2

        all_file_names.extend(file_names)
        all_local_indices.extend(local_indices)
        all_assignments.extend(assignments.cpu().detach())

        # save predictions so far
        if i == len_dataloader - 1 or (i > 0 and i % batch_div_amount == 0):
            save_idx = i // batch_div_amount
            print(f"{save_idx}, batch {i+1}/{len_dataloader}")
            save_indices.append(save_idx)
            save_file = f"{temp_save_path}/{save_idx}.npz"
            print(f"saving current predictions in {save_file}")
            np.savez(
                save_file,
                file_name=all_file_names,
                local_idx=all_local_indices,
                label=all_assignments,
            )
            all_file_names, all_local_indices, all_assignments = [], [], []
            print("saved current predictions")

    print("finished predicting")
    print("saving combined predictions")
    all_file_names, all_local_indices, all_labels = [], [], []
    for save_idx in save_indices:
        with np.load(f"{temp_save_path}/{save_idx}.npz") as assignment_info:
            all_file_names.extend(assignment_info["file_name"])
            all_local_indices.extend(assignment_info["local_idx"])
            all_labels.extend(assignment_info["label"])
    np.savez(
        f"{assignment_save_path}/{log_name}.npz",
        file_name=all_file_names,
        local_idx=all_local_indices,
        label=all_labels,
    )
    print("saved combined predictions")

    for save_idx in save_indices:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    print("removed temp files")


if __name__ == "__main__":
    main()
