import argparse
import os
import random as r

import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize

from src.categorisation.supervised.plaque_labels import (
    equidistant_labels,
    get_label_names,
)
from src.categorisation.visualisation import (
    add_label_names,
    label_incorrect_predictions,
)

r.seed(73)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ipc", "--images_per_cluster", default=10, type=int)
    parser.add_argument("-is", "--img_size", default=128, type=int)
    parser.add_argument("-ap", "--assignments_path", type=str)
    parser.add_argument(
        "-lif",
        "--labeled_idx_file",
        default="labeled_plaque_samples/labelfileidx.npz",
        type=str,
    )
    parser.add_argument("-sup", "--supervised", action="store_true", default=False)
    args = parser.parse_args()
    print(args)

    assignments_path = args.assignments_path
    img_per_cluster = args.images_per_cluster
    img_size = args.img_size
    labeled_idx_file = args.labeled_idx_file
    supervised = args.supervised

    label_names = get_label_names()

    model_name = assignments_path.split("/")[-1].split(".")[0]
    report_dir = f"result/class_report/{model_name}"

    with np.load(labeled_idx_file) as plaques_info:
        file_names = plaques_info["file_name"]
        local_indices = plaques_info["local_idx"]
        true_labels = plaques_info["label"]

    true_labels = equidistant_labels(true_labels)
    n_clusters = len(set(true_labels))

    image_matrix = np.zeros((img_per_cluster * img_size, n_clusters * img_size, 3))
    print("image matrix shape", image_matrix.shape)

    if supervised:
        found_labels = []
        predictions = []
        no_label_indicator = -2

    for cluster_idx in range(n_clusters):
        report_cluster_dir = f"{report_dir}/{cluster_idx+1}"
        cluster_files = os.listdir(report_cluster_dir)[:img_per_cluster]
        r.shuffle(cluster_files)
        for file_idx, file_name in enumerate(cluster_files):
            # read image from report dir
            img = imread(os.path.join(report_dir, str(cluster_idx + 1), file_name))
            img = resize(img, (img_size, img_size))
            # save image from report cluster directory in image matrix
            image_matrix[
                file_idx * img_size : file_idx * img_size + img_size,
                cluster_idx * img_size : cluster_idx * img_size + img_size,
            ] = img

            if supervised:
                wsi_file_name, wsi_local_idx = file_name.split("#")
                wsi_file_name = wsi_file_name.split(".")[0] + ".hdf5"
                wsi_local_idx = wsi_local_idx.split(".")[0]

                # if true label exists for current image, find it
                wsi_file_indices = np.where(file_names == wsi_file_name)[0]
                wsi_localidx_indices = np.where(
                    local_indices[wsi_file_indices] == wsi_local_idx
                )[0]
                img_label = true_labels[wsi_file_indices][wsi_localidx_indices]

                predictions.append(cluster_idx)
                if len(img_label):
                    if len(img_label) > 1:
                        raise Exception("Image belongs to multiple classes.")
                    found_labels.append(img_label[0])
                else:
                    # give impossible number in case image has no true label (unlabeled image)
                    found_labels.append(no_label_indicator)

    image_matrix = image_matrix * 255.0
    image_matrix = image_matrix.astype(np.uint8)

    # convert from numpy to pillow image
    image_matrix = Image.fromarray(image_matrix)

    if supervised:
        # not guaranteed to work if indices do not match with original files'
        # image_matrix = label_incorrect_predictions(image_matrix=image_matrix,
        #                                           predictions=predictions,
        #                                           true_labels=found_labels,
        #                                           label_names=label_names,
        #                                           img_size=img_size,
        #                                           no_label_indicator=no_label_indicator)
        image_matrix = add_label_names(image_matrix, label_names)

    image_matrix.save(os.path.join(report_dir, "image_matrix.png"))


if __name__ == "__main__":
    main()
