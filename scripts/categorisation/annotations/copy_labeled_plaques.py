import os

import h5py
import numpy as np

from src.util.cli import parse_config


# save a copy of the labeled plaques in a separate file
def copy_hdf5(src_file_path, dst_file_path, local_idx, group_name="plaques"):
    with h5py.File(src_file_path, "r") as src_file:
        with h5py.File(dst_file_path, "a") as dst_file:
            if group_name not in dst_file:
                dst_file.create_group(group_name)
            dst_file_group = dst_file[group_name]

            if local_idx in dst_file_group:
                del dst_file_group[local_idx]

            src_file.copy(f"{group_name}/{local_idx}", dst_file_group)


def main():
    config = parse_config()

    src_path = config["data_params"]["all_images_path"]
    dst_path = config["data_params"]["labeled_images_path"]
    labeled_idx_file = config["data_params"]["labeled_idx_file"]

    os.makedirs(dst_path, exist_ok=True)

    with np.load(labeled_idx_file) as plaques_info:
        file_names = plaques_info["file_name"]
        local_indices = plaques_info["local_idx"]

    unique_file_names = list(set(file_names))

    for filename in unique_file_names:
        matching_indices = np.where(file_names == filename)[0]
        m_local_indices = local_indices[matching_indices]

        for local_idx in m_local_indices:
            copy_hdf5(f"{src_path}/{filename}", f"{dst_path}/{filename}", local_idx)


if __name__ == "__main__":
    main()
