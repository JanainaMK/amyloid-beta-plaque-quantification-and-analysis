import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from src.categorisation.file_filter import valid_region
from src.categorisation.supervised.plaque_labels import get_label_names
from src.util.cli import parse_config


def annotate_thousand(number):
    return "{:,}".format(number)


def calc_relative_values(absolute_values):
    total = sum(absolute_values)
    if total == 0:
        return [0] * len(absolute_values)
    return [round((value / total) * 100, 2) for value in absolute_values]


def plaque_class_dist(wsi_dir, file_names, assignments):
    wsi_files = [
        os.path.splitext(wsi_f)[0].split(os.path.sep)[-1] + ".hdf5"
        for wsi_f in glob.glob(os.path.join(wsi_dir, "*.vsi"))
    ]

    selected_indices = [idx for idx, f in enumerate(file_names) if f in wsi_files]
    file_names = file_names[selected_indices]
    assignments = assignments[selected_indices]

    n_clusters = len(set(assignments))

    print(f"\nstats for {wsi_dir}:")

    print("number of files used:", len(set(file_names)))

    n_per_cluster = np.zeros(n_clusters, dtype=int)
    print("number of assigned plaques: ", len(assignments))
    for _, cluster in enumerate(assignments):
        # tracks number of plaques assigned to each cluster
        n_per_cluster[cluster] += 1

    print("number of clusters:", n_clusters)

    absolute_dist = []
    for i in range(n_clusters):
        abs_val = n_per_cluster[i]
        rel_val = round(n_per_cluster[i] * 100 / len(assignments), 2)
        print(
            f"cluster {i+1}:",
            annotate_thousand(abs_val),
            ",",
            rel_val,
            "% plaques assigned",
        )
        absolute_dist.append(abs_val)

    print("abs. dist.:", absolute_dist)

    return absolute_dist


def bar_plots(cent_abs_dist, ad_abs_dist, label_names, save_path):
    x_values = np.arange(len(label_names))
    width = 0.35
    _, ax = plt.subplots(figsize=(14, 8))

    # Plot bars
    bars1 = ax.bar(
        x_values - width / 2, cent_abs_dist, width, label="100+", color="skyblue"
    )
    bars2 = ax.bar(x_values + width / 2, ad_abs_dist, width, label="AD", color="salmon")

    ax.set_xlabel("Amyloid-Beta Types")
    ax.set_ylabel("Plaque Frequency")
    ax.set_title(
        f"Frequency of Plaques Across Aβ Types for\n100+ ({annotate_thousand(sum(cent_abs_dist))} Plaques) and AD ({annotate_thousand(sum(ad_abs_dist))} Plaques) Cohorts"
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(label_names)
    ax.legend()

    cent_rel_dist = calc_relative_values(cent_abs_dist)
    ad_rel_dist = calc_relative_values(ad_abs_dist)

    # Add text on top of each bar
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.2,
            f"{annotate_thousand(height)}\n({cent_rel_dist[i]}%)",
            ha="center",
            va="bottom",
        )

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.2,
            f"{annotate_thousand(height)}\n({ad_rel_dist[i]}%)",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    print(f"saved plot {save_path}")


def main():
    config = parse_config()

    assignments_path = config["params"]["assignments_path"]
    cent_dir = config["params"]["cent_dir"]
    ad_dir = config["params"]["ad_dir"]
    primary_labels = config["params"]["primary_labels"]
    plot_save_path = config["params"]["plot_save_path"]
    model_name = assignments_path.split("/")[-1].split(".")[0]
    plot_save_path += "/" + model_name
    os.makedirs(plot_save_path, exist_ok=True)

    with np.load(assignments_path) as plaques_info:
        file_names = plaques_info["file_name"]
        assignments = plaques_info["label"]

    selected_indices = [idx for idx, f in enumerate(file_names) if valid_region(f)]
    file_names = file_names[selected_indices]
    assignments = assignments[selected_indices]

    cent_abs_dist = plaque_class_dist(cent_dir, file_names, assignments)
    ad_abs_dist = plaque_class_dist(ad_dir, file_names, assignments)
    cent_abs_dist = np.array(cent_abs_dist)
    ad_abs_dist = np.array(ad_abs_dist)
    label_names = get_label_names()

    bar_plots(
        cent_abs_dist,
        ad_abs_dist,
        label_names,
        f"{plot_save_path}/plaque_freq_all_bar.png",
    )
    bar_plots(
        cent_abs_dist[primary_labels],
        ad_abs_dist[primary_labels],
        label_names[primary_labels],
        f"{plot_save_path}/plaque_freq_primary_bar.png",
    )


if __name__ == "__main__":
    main()
