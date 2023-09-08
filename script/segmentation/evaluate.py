import os
import argparse

import cv2
import h5py
import torch
import numpy as np

import src.segmentation.evaluation as eval
import src.segmentation.model as m
from src.data_access import DatasetReader
from src.util import LabelEnum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Segmentation evaluation')
parser.add_argument('model', type=str)
parser.add_argument('-ps', '--patch_size', default=1024, type=int)
parser.add_argument('-s', '--stride', default=-1, type=int)
parser.add_argument('-dl', '--downsample_level', default=16, type=int)
parser.add_argument('-p', '--partition', default='test', type=str)
args = parser.parse_args()

file = h5py.File('dataset/16x_set.hdf5', 'r')
patch_size = args.patch_size
stride = patch_size if args.stride == -1 else args.stride
downsample_lvl = args.downsample_level
label_type = LabelEnum.PIXEL
partition = args.partition
data_reader = DatasetReader(file[partition], patch_size, stride, downsample_lvl, label_type)
print('data ready:', data_reader.get_num_images(), 'images')

model_path = args.model
model_name = model_path.split('/')[1]
model = m.unet(model_path)
model.eval()
print('model ready:', model_name)

iou = np.zeros(data_reader.get_num_images())
dice = np.zeros(data_reader.get_num_images())
os.makedirs(f'evaluation/{model_name}', exist_ok=True)

with torch.no_grad():
    for i in range(data_reader.get_num_images()):
        print('evaluating', data_reader.image_list[i])
        image, label = data_reader.get_image_and_label(i)
        print('image:', i, ', image shape:', label.shape, ', patch shape:', image.patch_it.shape)
        label_tensor = torch.tensor(label.get_full()).detach().to(DEVICE)
        mask = eval.make_mask_patch_based(model, image)
        iou[i] = eval.intersection_over_union(mask, label_tensor).item()
        dice[i] = eval.dice(mask, label_tensor).item()
        np.save(f'evaluation/{model_name}/{model_name}--IoU.npy', iou)
        np.save(f'evaluation/{model_name}/{model_name}--DICE.npy', dice)
        print('IoU:', iou[i], ', DICE:', dice[i], '\n')

print('evaluation finished')


