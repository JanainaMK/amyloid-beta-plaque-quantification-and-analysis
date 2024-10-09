import argparse
import os

import h5py
from split_data import list_files, shuffle_data


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Link images and labels")
    parser.add_argument("-sd", "--save_dir", default="dataset", type=str)
    parser.add_argument("-rc", "--split_data", action="store_true", default=False)
    args = parser.parse_args()

    print("Settings: \nsave directory:", args.save_dir)

    return args


def link_images_labels(save_dir, linked_file_path):
    images_downsampled_dir = os.path.join(save_dir, "images")
    labels_downsampled_dir = os.path.join(save_dir, "labels")

    linked_file = h5py.File(linked_file_path, "w")
    try:
        init = "init"
        if init in linked_file:
            del init
            print(f"deleting old {init} group")

        init_group = linked_file.create_group(init)
        label_files = list_files(labels_downsampled_dir, "hdf5")

        print("\ninitializing image and label links")

        for label_file_path in label_files:
            label_name = os.path.splitext(label_file_path)[0].split(os.path.sep)[-1]

            images_downsampled_file_path = os.path.join(
                images_downsampled_dir, f"{label_name}.hdf5"
            )
            labels_downsampled_file_path = os.path.join(
                labels_downsampled_dir, f"{label_name}.hdf5"
            )

            if not os.path.exists(images_downsampled_file_path):
                raise FileNotFoundError(
                    f"A matching downsampled image file not found: '{images_downsampled_file_path}'"
                )

            if label_name in init_group:
                del init_group[label_name]
                print(f"deleted old {label_name} group")
            label_group = init_group.create_group(label_name)

            label_group["image_file"] = h5py.ExternalLink(
                images_downsampled_file_path, "/"
            )
            label_group["label_file"] = h5py.ExternalLink(
                labels_downsampled_file_path, "/"
            )

    except Exception as e:
        print(f"Error: {e}")
    finally:
        linked_file.close()


if __name__ == "__main__":
    args = parse_arguments()
    linked_file_path = os.path.join(args.save_dir, "linked_images_labels.hdf5")

    link_images_labels(args.save_dir, linked_file_path)
    if args.split_data:
        with h5py.File(linked_file_path, "a") as linked_file:
            shuffle_data(linked_file)
