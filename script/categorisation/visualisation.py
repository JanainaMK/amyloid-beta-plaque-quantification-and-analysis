import os
import argparse

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument('report_dir', type=str)
parser.add_argument('-noc', '--number_of_clusters', type=int)
parser.add_argument('-ipc', '--images_per_cluster', type=int)
parser.add_argument('-is', '--img_size', default=100, type=int)
args = parser.parse_args()
print(args)

report_dir = args.report_dir
n_clusters = args.number_of_clusters
img_per_cluster = args.images_per_cluster
img_size = args.img_size

name = report_dir.split('/')[-1]
res = np.zeros((img_per_cluster * img_size, (n_clusters + 1) * img_size, 3))

for i in range(n_clusters + 1):
    for j, filename in enumerate(os.listdir(os.path.join(report_dir, str(i)))):
        img = imread(os.path.join(report_dir, str(i), filename))
        img = resize(img, (img_size, img_size))
        res[j*img_size:j*img_size + img_size, i*img_size:i*img_size + img_size] = img

imsave(os.path.join(report_dir, f'{name}.png'), res)
