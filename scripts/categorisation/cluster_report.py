import argparse
import random
import os

import numpy as np
import h5py
from skimage.io import imsave


parser = argparse.ArgumentParser()
parser.add_argument('assignment', type=str)
parser.add_argument('-noi', '--number_of_images', default=10, type=int)
args = parser.parse_args()
print(args)

assignment_path = args.assignment
n_img = args.number_of_images

name = assignment_path.split('/')[-1][:-4]
info = name.split('-')
pre_ass_path = 'result/cluster-assignment'
assignment = np.load(os.path.join(pre_ass_path, assignment_path))

n_clusters = assignment.max() + 1
n_per_cluster = np.zeros(n_clusters + 1, dtype=int)
indices_per_cluster = [[] for i in range(n_clusters + 1)]
print('number of assigned plaques: ', len(assignment))
for i, c in enumerate(assignment):
    # tracks number of plaques assigned to each cluster
    n_per_cluster[c + 1] += 1
    # tracks which plaques (indices) are assigned to each cluster
    indices_per_cluster[c + 1].append(i)

print('number of clusters:', n_clusters)
for i in range(1, len(n_per_cluster)):
    print(f'cluster {i}:', n_per_cluster[i], 'plaques assigned')

image_cluster_dict = {}  # image index: cluster index
example_indices = set()
for i, indices in enumerate(indices_per_cluster):
    if i != 0:
        os.makedirs(f'result/cluster-report/{name}/{i}', exist_ok=True)
    # randomly shuffles the plaque indices for a cluster
    random.shuffle(indices)
    for j in indices[:n_img]:
        # tracks plaque image indices sampled from random shuffle
        example_indices.add(j)
        # tracks cluster assigned to each sampled plaque image
        image_cluster_dict[j] = i
i = 0
for filename in os.listdir('result/features'):
    file = h5py.File(f'result/features/{filename}', 'r')
    for plaque_name in file['plaques']:
        if i in example_indices:
            os.makedirs(f'result/cluster-report/{name}/{image_cluster_dict[i]}/', exist_ok=True)
            imsave(
                f'result/cluster-report/{name}/{image_cluster_dict[i]}/{i}.png',
                file[f'plaques/{plaque_name}/plaque'][()],
            )
        i += 1






