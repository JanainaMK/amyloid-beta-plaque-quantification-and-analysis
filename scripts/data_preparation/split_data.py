import glob
import os
import random

import h5py
from sklearn.model_selection import train_test_split


def shuffle_data(file: h5py.File, train_size=0.6, validation_size=0.2, random_state=42):
    print("\nassigning shuffled data to train, validation, and test set")

    key_list = [key for key in file["init"].keys()]
    random.shuffle(key_list)
    # Split the data into train, test, and validation sets
    train_names, temp_names = train_test_split(
        key_list, train_size=train_size, random_state=random_state
    )
    test_names, val_names = train_test_split(
        temp_names, test_size=validation_size, random_state=random_state
    )
    for name in train_names:
        file.move(f"/init/{name}", f"/train/{name}")
    for name in test_names:
        file.move(f"/init/{name}", f"/test/{name}")
    for name in val_names:
        file.move(f"/init/{name}", f"/validation/{name}")


def reset_data(group: h5py.Group):
    for partition in group:
        if partition == "init":
            continue
        reset_data_in_partition(group, partition)


def reset_data_in_partition(group: h5py.Group, partition: str):
    for image_group_name in group[partition]:
        group.move(f"/{partition}/{image_group_name}", f"/init/{image_group_name}")


def move_random(file: h5py.File, target: str, n: int):
    keylist = [key for key in file["init"].keys()]
    indices = random.sample(range(len(keylist)), n)
    for i in indices:
        key = keylist[i]
        file.move(f"/init/{key}", f"/{target}/{key}")


def list_files(dir, extension):
    file_path = os.path.join(dir, f"*.{extension}")
    return [f for f in glob.glob(file_path)]
