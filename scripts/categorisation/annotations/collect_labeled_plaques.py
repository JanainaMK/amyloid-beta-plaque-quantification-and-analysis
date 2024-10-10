import os
import shutil

import numpy as np
import pandas as pd
from torchvision.transforms import ToPILImage

from src.categorisation.cli import parse_config
from src.categorisation.plaques_dataset import PlaquesDataset
from src.categorisation.supervised.plaque_labels import equidistant_labels


def collect_labeled_plaques(
    dataloader,
    n_samples: int,
    n_cols: int,
    n_rows: int,
    save_dir: str,
    samples_save_dir: str,
    labeled_vis_save_dir: str,
    labeled_plaques_save_file: str,
    labeled_plaques_read_file: str,
    label_names_read_file: str,
    display_plaques: bool = True,
    **kwargs,
):
    n_image_matrices = n_samples
    print("number of samples:", n_image_matrices)
    print("number of columns for each sample:", n_cols)
    print("number of rows for each sample:", n_rows, "\n")

    collected_labels = []
    collected_file_names = []
    collected_local_idxs = []

    image_matrix_idx = 0
    row_idx = 0
    col_idx = 0

    # all paths will have save_dir as parent
    samples_save_dir = save_dir + "/" + samples_save_dir
    labeled_vis_save_dir = save_dir + "/" + labeled_vis_save_dir
    labeled_plaques_save_file = save_dir + "/" + labeled_plaques_save_file
    labeled_plaques_read_file = save_dir + "/" + labeled_plaques_read_file
    label_names_read_file = save_dir + "/" + label_names_read_file

    # read in csv data
    label_names = pd.read_csv(label_names_read_file).drop_duplicates()

    labeled_plaque_info = pd.read_csv(labeled_plaques_read_file).drop_duplicates()
    csv_columns = [
        "Label",
        "Sample",
        "Row",
        "Column",
    ]  # assumes these column names compose csv
    labeled_plaque_info = labeled_plaque_info[csv_columns]
    labeled_plaque_info = labeled_plaque_info.dropna()
    labeled_plaque_info = labeled_plaque_info.astype(int)

    lpi_temp = labeled_plaque_info[labeled_plaque_info["Sample"] == image_matrix_idx]

    # create directory for each class
    if display_plaques:
        for label_name in list(set(label_names["Name"])):
            img_save_path = os.path.join(labeled_vis_save_dir, label_name)
            if os.path.exists(img_save_path):
                shutil.rmtree(img_save_path)
            os.makedirs(img_save_path, exist_ok=True)

    for batch in dataloader:
        if n_image_matrices <= 0:
            return
        imgs, batch_data = batch

        for didx in range(len(imgs)):
            # if number of rows is reached, move on to next column
            if row_idx >= n_rows:
                row_idx = 0
                col_idx += 1

            # if number of columns is reached, move on to next image matrix
            if col_idx >= n_cols:

                image_matrix_idx += 1

                if image_matrix_idx >= n_image_matrices:
                    # if sample number is reached, end method
                    break
                else:
                    # if sample number is not reached, continue creating next image matrix sample
                    row_idx = 0
                    col_idx = 0

            lpi_temp = labeled_plaque_info[
                labeled_plaque_info["Sample"] == image_matrix_idx
            ]

            labeled_plaque = lpi_temp[
                (lpi_temp["Row"] == row_idx) & (lpi_temp["Column"] == col_idx)
            ]
            # if a labeled plaque exists, collect its label
            if len(labeled_plaque) > 0:
                label = labeled_plaque["Label"]
                if len(label) > 1:
                    raise Exception(
                        "The same plaque is assigned to more than one class."
                    )
                label = label.iloc[0]
                file_names = batch_data[0]
                local_indices = batch_data[1]

                collected_labels.append(label)
                collected_file_names.append(file_names[didx])
                collected_local_idxs.append(local_indices[didx])

                # save labeled plaque images separately for visualisation purposes
                if display_plaques:
                    label_name = label_names[label_names["Value"] == label]["Name"]
                    if len(label_name) > 1:
                        raise Exception(
                            "The same class name is assigned to more than one class."
                        )
                    label_name = label_name.iloc[0]

                    img = imgs[didx]
                    to_pil = ToPILImage()
                    img = to_pil(img)
                    img.save(
                        os.path.join(
                            labeled_vis_save_dir,
                            label_name,
                            f"s{image_matrix_idx}_r{row_idx}_c{col_idx}.png",
                        )
                    )

            row_idx += 1

        if image_matrix_idx >= n_image_matrices:
            break

    collected_labels = equidistant_labels(collected_labels)
    np.savez(
        labeled_plaques_save_file,
        label=collected_labels,
        file_name=collected_file_names,
        local_idx=collected_local_idxs,
    )
    print(f"collected and saved {len(collected_labels)} labeled plaques")


def main():
    config = parse_config()

    datamodule = PlaquesDataset(
        **config["data_params"], pin_memory=len(config["run_params"]["gpus"]) != 0
    )
    datamodule.setup()

    dataloader = datamodule.train_dataloader()
    collect_labeled_plaques(
        dataloader=dataloader, display_plaques=True, **config["samples_param"]
    )


if __name__ == "__main__":
    main()
