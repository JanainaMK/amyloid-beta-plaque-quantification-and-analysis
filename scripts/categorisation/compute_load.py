import os

import h5py
import numpy as np

from src.categorisation.file_filter import (
    get_dataset_name,
    get_participant_number,
    get_region_name,
)
from src.categorisation.cli import parse_config


def main():
    config = parse_config()

    all_images_path = config["params"]["all_images_path"]
    cent_dir = config["params"]["cent_dir"]
    ad_dir = config["params"]["ad_dir"]
    assignments_path = config["params"]["assignments_path"]
    load_file = config["params"]["load_file"]

    load_save_dir = os.path.dirname(load_file)
    os.makedirs(load_save_dir, exist_ok=True)

    total_plaques = 0
    regions = []
    datasets = []
    loads = []
    predictions = []
    participants = []

    with np.load(assignments_path, allow_pickle=True) as plaques_info:
        saved_file_names = plaques_info["file_name"]
        saved_local_indices = plaques_info["local_idx"]
        saved_assignments = plaques_info["label"]

    n_classes = len(set(saved_assignments))
    n_files = len(set(saved_file_names))
    print("number of classes:", n_classes)
    print("number of files:", n_files)

    for i, image_name in enumerate(set(saved_file_names)):
        print("slide", i + 1, f"/{n_files}:", image_name)

        wsi_indices = np.where(saved_file_names == image_name)[0]
        plaque_indices = saved_local_indices[wsi_indices]
        plaque_predictions = saved_assignments[wsi_indices]

        image_name = os.path.splitext(image_name)[0]

        try:
            with h5py.File(f"{all_images_path}/{image_name}.hdf5", "r") as result_file:
                gm = result_file["grey-matter"][()]
                total_area = np.sum(gm) * (0.274 * 16) ** 2
                area_per_class = {}
                # save load per class for the file
                for i in range(n_classes):
                    area_per_class[i] = 0
                for pi_idx, pi in enumerate(plaque_indices):
                    plaque = result_file["plaques"][f"{pi}"]
                    pred = plaque_predictions[pi_idx]
                    area_per_class[pred] += plaque.attrs["area"]

                region_name = get_region_name(image_name)
                dataset_name = get_dataset_name(image_name, cent_dir, ad_dir)
                participant = get_participant_number(image_name)
                for i in range(n_classes):
                    load = (area_per_class[i] / total_area) * 100  # in percentage
                    predictions.append(i)
                    loads.append(load)
                    regions.append(region_name)
                    datasets.append(dataset_name)
                    participants.append(participant)

                total_plaques += result_file.attrs["n_plaques"]
        except FileNotFoundError as e:
            raise FileNotFoundError(e)
        finally:
            result_file.close()

    print("number of plaques:", total_plaques)
    print("number of ab loads:", len(loads))

    np.savez(
        load_file,
        region=regions,
        dataset=datasets,
        load=loads,
        label=predictions,
        participant=participants,
    )


if __name__ == "__main__":
    main()
