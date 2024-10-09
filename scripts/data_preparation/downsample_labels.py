import argparse
import os

import cv2 as cv
import h5py
import numpy as np
from split_data import list_files


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Downsample label")
    parser.add_argument("-dl", "--downsample_level", default=16, type=int)
    parser.add_argument("-sd", "--save_dir", default="dataset/", type=str)
    parser.add_argument(
        "-ld",
        "--labels_dir",
        default=os.path.join("dataset", "grey matter labels"),
        type=str,
    )
    parser.add_argument("-rc", "--replace_created", action="store_true", default=True)
    args = parser.parse_args()

    print(
        "Settings:",
        f"\n\tdownsample level: {args.downsample_level}x",
        "\nsave directory:",
        args.save_dir,
        "\nlabels directory:",
        args.labels_dir,
    )

    return args


def save_downsampled_labels(args):
    """
    Save downsampled labels.
    """
    images_target_dir = "images"
    labels_target_dir = "labels"
    downsample_lvl = args.downsample_level
    replace_created = args.replace_created
    labels_dir = args.labels_dir

    images_downsampled_dir = os.path.join(args.save_dir, images_target_dir)
    labels_downsampled_dir = os.path.join(args.save_dir, labels_target_dir)

    os.makedirs(labels_downsampled_dir, exist_ok=True)
    print(f"\nsaving {downsample_lvl}x downsampled labels")

    label_files = list_files(labels_dir, "hdf5")
    for i, label_file_path in enumerate(label_files):
        label_name = os.path.splitext(label_file_path)[0].split(os.path.sep)[-1]

        images_downsampled_file_path = os.path.join(
            images_downsampled_dir, f"{label_name}.hdf5"
        )
        labels_downsampled_file_path = os.path.join(
            labels_downsampled_dir, f"{label_name}.hdf5"
        )

        if replace_created or not os.path.exists(labels_downsampled_file_path):
            print(f"{i+1}/{len(label_files)} files, {label_name}")
            if not os.path.exists(images_downsampled_file_path):
                raise FileNotFoundError(
                    f"A matching downsampled image file was not found: '{images_downsampled_file_path}'"
                )

            with h5py.File(images_downsampled_file_path, "r") as image_file:
                image_shape = image_file[f"{downsample_lvl}x"].shape

            # Read label file
            with h5py.File(label_file_path, "r") as label_file:
                label = label_file["grey matter labels"][()].astype(np.uint8) * 255

            # Downsample label
            label = cv.resize(label, (image_shape[2], image_shape[1]))
            _, label = cv.threshold(label, 127, 255, cv.THRESH_BINARY)

            with h5py.File(labels_downsampled_file_path, "w") as ds_file:
                ds_file.create_dataset(f"{downsample_lvl}x", data=label.astype(bool))


if __name__ == "__main__":
    args = parse_arguments()

    save_downsampled_labels(args)
