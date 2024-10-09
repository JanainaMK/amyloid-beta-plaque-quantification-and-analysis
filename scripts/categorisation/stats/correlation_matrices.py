import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from src.categorisation.stats import get_significance_level
from src.categorisation.supervised.plaque_labels import get_label_names
from src.util.cli import parse_config


def plot_corr_matrix(
    orig_df, label_names, variables, variable_names, plot_name, plot_save_path
):
    if len(variables) == 0:
        return
    corr_matrix = pd.DataFrame()
    annotation_matrix = pd.DataFrame()
    use_region = False
    df = orig_df

    # Check if current variables refer to cerebral regions
    if df["Region"].isin(variable_names).any():
        use_region = True

    for label_name in label_names:
        for var_idx, var in enumerate(variables):
            if use_region:
                # use only relevant region
                df = orig_df[orig_df["Region"] == variable_names[var_idx]]
            corr, p_value = spearmanr(df[label_name], df[var])
            corr = round(corr, 2)
            corr_matrix.loc[label_name, var] = corr
            annotation_matrix.loc[label_name, var] = (
                str(corr) + "\n" + get_significance_level(p_value)
            )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        corr_matrix,
        annot=annotation_matrix,
        fmt="",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        xticklabels=variable_names,
    )
    plt.xticks(rotation=0)
    plt.title(f"Correlation Matrix between Predicted \nAβ Type Loads and {plot_name}")
    plt.tight_layout()
    plt.savefig(f"{plot_save_path}/{plot_name}_corr_matrix.png", dpi=600)


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

    variables = [
        [
            "MMSE",
            "Clockdrawing test",
            "Digitspan forward",
            "Digitspan backward",
            "Key Search",
        ],
        ["Thal_AB_phase", "Thal_CAA_stage", "CERAD_NP"],
        ["AB_frontal", "AB_lpi", "AB_occipital", "AB_temporal"],
    ]
    names = [
        ["MMSE", "CDT", "DSF", "DSB", "KS"],
        ["Thal AB phase", "Thal CAA stage", "CERAD NP"],
        ["Frontal", "Parietal", "Occipital", "Temporal"],
    ]
    plot_names = [
        "Cognitive Tests",
        "Neuropathological Staging Schemes",
        "Cerebral Regions",
    ]

    for idx, var in enumerate(variables):
        plot_corr_matrix(
            merged_df, label_names, var, names[idx], plot_names[idx], plot_save_path
        )


if __name__ == "__main__":
    main()
