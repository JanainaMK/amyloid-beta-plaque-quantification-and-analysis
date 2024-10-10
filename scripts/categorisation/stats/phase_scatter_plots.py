import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from src.categorisation.stats import get_significance_level
from src.categorisation.supervised.plaque_labels import get_label_names
from src.categorisation.cli import parse_config


def main():
    config = parse_config()

    load_file = config["params"]["load_file"]
    primary_labels = config["params"]["primary_labels"]
    plot_save_path = config["params"]["plot_save_path"]
    participant_data_file = config["params"]["participant_data_file"]
    model_name = load_file.split("/")[-1].split(".")[0]
    plot_save_path += "/" + model_name
    os.makedirs(plot_save_path, exist_ok=True)

    with np.load(load_file) as saved_data:
        saved_regions = saved_data["region"]
        saved_datasets = saved_data["dataset"]
        saved_loads = saved_data["load"]
        saved_predictions = saved_data["label"]
        saved_participants = saved_data["participant"]

    label_names = get_label_names()

    named_predictions = [label_names[label] for label in saved_predictions]
    label_names = label_names[primary_labels]

    df = pd.DataFrame(
        {
            "Region": saved_regions,
            "Dataset": saved_datasets,
            "Load": saved_loads,
            "Prediction": named_predictions,
            "NBB": saved_participants,
        }
    )
    df = df[df["Prediction"].isin(label_names)]

    df_pivot = df.pivot_table(
        index=["NBB", "Region", "Dataset"],
        columns="Prediction",
        values="Load",
        aggfunc="first",
    ).reset_index()

    csv_df = pd.read_csv(participant_data_file).drop_duplicates()
    csv_df = csv_df.dropna()

    # merge ab load and participant data
    merged_df = pd.merge(csv_df, df_pivot, on=["NBB"], how="inner").reset_index()

    plot_variables = [
        ["CAA", "Thal_CAA_stage", "Thal CAA Stage"],
        ["Cored", "CERAD_NP", "CERAD NP"],
    ]

    h = 0.02
    spacing = 0.05
    n_stages = 4
    for var_y, var_x, name in plot_variables:
        plt.figure(figsize=(10, 6))
        plt.scatter(merged_df[var_x], merged_df[var_y], color="steelblue")

        for stage in range(n_stages):
            sns.boxplot(
                x=merged_df[var_x][merged_df[var_x] == stage],
                y=merged_df[var_y][merged_df[var_x] == stage],
                color="black",
                medianprops={"color": "red"},  # color median red
                fliersize=0,
                width=0.2,
                fill=False,
            )

            # plot significance level along with line
            if stage < n_stages - 1:
                # measure significance between stages
                group1 = merged_df[var_y][merged_df[var_x] == stage]
                group2 = merged_df[var_y][merged_df[var_x] == stage + 1]
                _, p = mannwhitneyu(group1, group2)

                # significance line coordinates
                x1, x2 = stage, stage + 1 - spacing
                y_max = merged_df[var_y].max() + 0.5
                plt.plot(
                    [x1, x1, x2, x2],
                    [y_max, y_max + h, y_max + h, y_max],
                    lw=1,
                    color="black",
                )
                plt.text(
                    (x1 + x2) / 2,
                    y_max + h,
                    get_significance_level(p),
                    ha="center",
                    va="bottom",
                    color="black",
                )

        plt.title(f"Predicted {var_y} Aβ Loads over {name} Scores")
        plt.xlabel(f"{name} Scores")
        plt.ylabel(f"Predicted {var_y} Aβ Load (%)")
        plt.tight_layout()
        plt.savefig(f"{plot_save_path}/{var_y}_{name}_scatter.png", dpi=600)


if __name__ == "__main__":
    main()
