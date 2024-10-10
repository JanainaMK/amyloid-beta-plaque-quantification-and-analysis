import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from src.categorisation.cli import parse_config
from src.categorisation.stats import get_significance_level
from src.categorisation.supervised.plaque_labels import get_label_names


# print more detailed stats for exact values
def print_stats(df):
    df = df.groupby(["Prediction", "Dataset"])
    stats = []
    for pd_values, grouped_data in df:
        prediction, dataset = pd_values
        median = grouped_data["Load"].median()
        q1 = grouped_data["Load"].quantile(0.25)
        q3 = grouped_data["Load"].quantile(0.75)
        iqr = q3 - q1

        stats.append(
            {
                "Prediction": prediction,
                "Dataset": dataset,
                "Median": median,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
            }
        )
    print(stats)


def plot_significance(df, names, p_values, h):
    # Annotate statistical significance with '*'
    fontsize = 14
    for i, name in enumerate(names):
        x1, x2 = i - 0.2, i + 0.2
        y_max = df["Load"].max()
        plt.plot(
            [x1, x1, x2, x2], [y_max, y_max + h, y_max + h, y_max], lw=1.5, c="black"
        )
        p = p_values[name]
        plt.text(
            (x1 + x2) / 2,
            y_max + h,
            get_significance_level(p),
            ha="center",
            va="bottom",
            color="black",
            fontsize=fontsize,
        )


def violin_across_classes(
    df, label_names, dataset_names, color_palette, save_path, display_stats=False
):
    plt.figure(figsize=(10, 6))

    # Box plot with different datasets as subgroups
    sns.boxplot(
        data=df,
        x="Prediction",
        y="Load",
        hue="Dataset",
        width=0.3,
        linewidth=1.5,
        fliersize=0,
        order=label_names,
        hue_order=dataset_names,
        palette=color_palette,
    )

    p_values = {}
    for label_name in label_names:
        # measure significance between datasets for each prediction
        group_x = df[
            (df["Prediction"] == label_name) & (df["Dataset"] == dataset_names[0])
        ]["Load"]
        group_y = df[
            (df["Prediction"] == label_name) & (df["Dataset"] == dataset_names[1])
        ]["Load"]
        _, p_val = mannwhitneyu(group_x, group_y)
        p_values[label_name] = p_val

    plot_significance(df, label_names, p_values, 0.1)

    if display_stats:
        print_stats(df)

    plt.legend()
    plt.xlabel("Aβ Types")
    plt.ylabel("Aβ Load (%)")
    plt.title("Aβ Load across Different Aβ Plaque Types")
    plt.tight_layout()
    plt.savefig(f"{save_path}/load_across_classes.png", dpi=600)


def violin_across_regions_per_class(
    orig_df,
    label_names,
    dataset_names,
    region_names,
    color_palette,
    save_path,
    display_stats=False,
):
    heights = [0.09, 0.015, 0.01, 0.008, 0.005, 0.005]  # for plotting '*' text

    for label_idx in range(len(label_names)):
        df = orig_df[orig_df["Prediction"] == label_names[label_idx]]

        plt.figure(figsize=(8, 6))

        # Violin plots
        sns.violinplot(
            data=df,
            x="Region",
            y="Load",
            hue="Dataset",
            split=True,
            inner="quartile",
            order=region_names,
            hue_order=dataset_names,
            palette=color_palette,
            cut=0,
        )

        # Box plots
        sns.boxplot(
            data=df,
            x="Region",
            y="Load",
            hue="Dataset",
            width=0.3,
            linewidth=1.5,
            fliersize=0,
            order=region_names,
            hue_order=dataset_names,
            palette=color_palette,
        )

        p_values = {}
        for region_name in region_names:
            # measure significance between datasets for each region
            group_x = df[
                (df["Region"] == region_name) & (df["Dataset"] == dataset_names[0])
            ]["Load"]
            group_y = df[
                (df["Region"] == region_name) & (df["Dataset"] == dataset_names[1])
            ]["Load"]
            _, p_val = mannwhitneyu(group_x, group_y)
            p_values[region_name] = p_val

        plot_significance(df, region_names, p_values, heights[label_idx])

        if display_stats:
            print_stats(df)

        # Plot legend
        handles, _ = plt.gca().get_legend_handles_labels()
        plt.legend(handles, dataset_names, loc="upper right")

        # Remove legend
        # plt.legend().remove()

        plt.xlabel("Cerebral Regions")
        plt.ylabel("Aβ Load (%)")
        plt.title(f"{label_names[label_idx]} Aβ Load across Different Cerebral Regions")
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/{label_names[label_idx]}_load_across_regions_violin.png",
            dpi=600,
        )


def main():
    config = parse_config()

    primary_labels = config["params"]["primary_labels"]
    plot_save_path = config["params"]["plot_save_path"]
    load_file = config["params"]["load_file"]
    model_name = load_file.split("/")[-1].split(".")[0]
    plot_save_path += "/" + model_name

    os.makedirs(plot_save_path, exist_ok=True)

    with np.load(load_file) as saved_data:
        saved_regions = saved_data["region"]
        saved_datasets = saved_data["dataset"]
        saved_loads = saved_data["load"]
        saved_predictions = saved_data["label"]

    label_names = get_label_names()
    named_predictions = [label_names[label] for label in saved_predictions]
    label_names = label_names[primary_labels]  # use only primary labels

    dataset_names = ["100+", "AD"]
    region_names = ["Frontal", "Parietal", "Temporal", "Occipital"]

    color_palette = {
        "100+": "#2cb8f0",  # blue shade
        "AD": "#f75343",  # red shade
    }

    df = pd.DataFrame(
        {
            "Region": saved_regions,
            "Dataset": saved_datasets,
            "Load": saved_loads,
            "Prediction": named_predictions,
        }
    )
    df = df[df["Prediction"].isin(label_names)]  # keep only primaries

    violin_across_classes(df, label_names, dataset_names, color_palette, plot_save_path)
    violin_across_regions_per_class(
        df, label_names, dataset_names, region_names, color_palette, plot_save_path
    )


if __name__ == "__main__":
    main()
