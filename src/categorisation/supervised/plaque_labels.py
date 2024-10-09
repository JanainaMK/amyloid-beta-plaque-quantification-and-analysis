import numpy as np
import pandas as pd


# count how many samples are used for a class
def count_occurrences(labels):
    label_occurences = {}
    for label in labels:
        if label in label_occurences:
            label_occurences[label] += 1
        else:
            label_occurences[label] = 1

    sorted_numbers = sorted(label_occurences.keys())
    for label in sorted_numbers:
        count = label_occurences[label]
        print(f"{label}: {count} times, {count/len(labels)}")
    return label_occurences


# adjust the labels such that the new labels maintain the original order but now are evenly spaced
def equidistant_labels(labels):
    distinct_labels = sorted(set(labels))
    new_labels = []

    for label in labels:
        new_label = distinct_labels.index(int(label))
        new_labels.append(new_label)

    return np.array(new_labels)


# get label names saved in csv file
def get_label_names(
    num_classes=9, label_names_file="labeled_plaque_samples/label_names.csv"
):
    label_name_data = pd.read_csv(label_names_file).drop_duplicates()
    label_name_data = label_name_data.sort_values(by="Value")
    label_names = label_name_data["Name"].tolist()

    if num_classes != len(label_names):
        print(f"number of label names: {len(label_names)}")
        raise Exception("Number of classes does not match number of label names.")

    return np.array(label_names)
