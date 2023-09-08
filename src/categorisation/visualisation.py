import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from skimage.io import imsave, ImageCollection

ACTIVE_PATH = 'visuals/cluster/active_clustering'


# class DendrogramLabelMaker:
#     def __init__(self, label_dict=None):
#         self.label_dict = label_dict
#
#     def generate_label_dict(self, R):
#         temp = {R["leaves"][ii]: labels[ii] for ii in range(len(R["leaves"]))}
#
#     def get_label(self, item):
#         return self.labeldict[item]


def plot_dendrogram(model: AgglomerativeClustering, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix,**kwargs)


def make_cluster_folders(images: ImageCollection, assignments: np.ndarray, n_clusters: int, target: str = ACTIVE_PATH):
    if os.path.exists(target):
        shutil.rmtree(target)
    os.mkdir(target)
    for i in range(-1, n_clusters):
        os.mkdir(os.path.join(target, f'cluster_{i}'))
    for i, image in enumerate(images):
        path = os.path.join(target, f'cluster_{assignments[i]}', f'{i}.png')
        imsave(path, image)


