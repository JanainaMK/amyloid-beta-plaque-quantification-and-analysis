import argparse
import os
import random as r
import shutil

import h5py
import numpy as np
from skimage.io import imsave

from src.categorisation.file_filter import valid_region

r.seed(23)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-noi", "--number_of_images", default=10, type=int)
    parser.add_argument("-ip", "--images_path", type=str)
    parser.add_argument("-ap", "--assignments_path", type=str)
    args = parser.parse_args()
    print(args)

    n_img = args.number_of_images

    assignments_path = args.assignments_path
    images_path = args.images_path
    model_name = assignments_path.split("/")[-1].split(".")[0]

    with np.load(assignments_path) as plaques_info:
        file_names = plaques_info["file_name"]
        local_indices = plaques_info["local_idx"]
        assignments = plaques_info["label"]

    selected_indices = [idx for idx, f in enumerate(file_names) if valid_region(f)]
    file_names = file_names[selected_indices]
    local_indices = local_indices[selected_indices]
    assignments = assignments[selected_indices]

    n_clusters = len(set(assignments))

    print("number of files used:", len(set(file_names)))

    n_per_cluster = np.zeros(n_clusters, dtype=int)
    indices_per_cluster = [[] for _ in range(n_clusters)]
    print("number of assigned plaques: ", len(assignments))
    for i, cluster in enumerate(assignments):
        # tracks number of plaques assigned to each cluster
        n_per_cluster[cluster] += 1
        # tracks which plaques (indices) are assigned to each cluster
        indices_per_cluster[cluster].append(i)

    print("number of clusters:", n_clusters)
    for i in range(n_clusters):
        abs_val = n_per_cluster[i]
        rel_val = round(n_per_cluster[i] * 100 / len(assignments), 2)
        print(f"cluster {i+1}:", abs_val, ",", rel_val, "% plaques assigned")

    report_path = "result/class_report"
    if os.path.exists(f"{report_path}/{model_name}"):
        shutil.rmtree(f"{report_path}/{model_name}")
    for cluster, indices in enumerate(indices_per_cluster):
        cluster_report_path = f"{report_path}/{model_name}/{cluster+1}/"
        os.makedirs(cluster_report_path, exist_ok=True)
        # randomly shuffles the plaque indices for a cluster
        r.shuffle(indices)

        # add images to cluster
        for idx in indices[:n_img]:
            file_name = file_names[idx]
            local_plaque_index = local_indices[idx]
            with h5py.File(f"{images_path}/{file_name}", "r") as file:
                image = file[f"plaques/{local_plaque_index}/plaque"][()]
                file_name = file_name.split(".")[0]
                imsave(
                    f"{cluster_report_path}/{file_name}#{local_plaque_index}.png",
                    image,
                )


if __name__ == "__main__":
    main()
