import os
import argparse

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument('report_dir', type=str)
parser.add_argument('-noc', '--number_of_clusters', default=20, type=int)
parser.add_argument('-ipc', '--images_per_cluster', default=10, type=int)
parser.add_argument('-is', '--img_size', default=100, type=int)
args = parser.parse_args()
print(args)

report_dir = args.report_dir
n_clusters = args.number_of_clusters
img_per_cluster = args.images_per_cluster
img_size = args.img_size

name = report_dir.split('/')[-1]
res = np.zeros((img_per_cluster * img_size, n_clusters * img_size, 3))
print(res.shape)

for i in range(1, n_clusters + 1):
    report_cluster_dir = os.path.join(report_dir, str(i))
    for j, filename in enumerate(os.listdir(report_cluster_dir)[:img_per_cluster]):
        img = imread(os.path.join(report_dir, str(i), filename))
        img = resize(img, (img_size, img_size))
        # save image from report cluster directory in image matrix
        res[j*img_size:j*img_size + img_size, i*img_size - img_size:i*img_size] = img
        
res = res * 255.0  # Scale to 0-255 range
res = res.astype(np.uint8)
imsave(os.path.join(report_dir, f'{name}.png'), res)
