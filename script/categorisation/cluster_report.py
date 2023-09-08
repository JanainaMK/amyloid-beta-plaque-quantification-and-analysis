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
n_per_cluster = np.zeros(n_clusters + 1)
indices_per_cluster = [[] for i in range(n_clusters + 1)]
print(indices_per_cluster)
for i, c in enumerate(assignment):
    n_per_cluster[c + 1] += 1
    indices_per_cluster[c + 1].append(i)

print('number of clusters:', n_clusters)
print('outliers:', n_per_cluster[0], len(indices_per_cluster[0]))
for i in range(1, len(n_per_cluster)):
    print(f'cluster {i}:', n_per_cluster[i], len(indices_per_cluster[i]))

image_cluster_dict = {}  # image index: cluster index
example_indices = []
for i, indices in enumerate(indices_per_cluster):
    os.mkdir(f'result/cluster-report/{name}/{i}')
    random.shuffle(indices)
    for j in indices[:n_img]:
        example_indices.append(j)
        image_cluster_dict[j] = i

print(image_cluster_dict)

i = 0
for filename in os.listdir('result/images'):
    file = h5py.File(f'result/images/{filename}', 'r')
    for plaque_name in file['plaques']:
        if i in example_indices:
            imsave(
                f'result/cluster-report/{name}/{image_cluster_dict[i]}/{i}.png',
                file[f'plaques/{plaque_name}/plaque'][()],
            )
        i += 1






