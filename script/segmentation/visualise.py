import argparse

import cv2
import h5py
import torch
import numpy as np

import src.segmentation.evaluation as eval
import src.segmentation.model as m
from src.data_access import ImageReader

parser = argparse.ArgumentParser(description='Segmentation evaluation')
parser.add_argument('image', type=str)
parser.add_argument('-m', '--model', default='models/2023-03-15-unet-16x-bs16-ps256-lr0.0001/2023-03-15-unet-16x-bs16-ps256-lr0.0001-e3v49.pt', type=str)
parser.add_argument('-ps', '--patch_size', default=1024, type=int)
parser.add_argument('-s', '--stride', default=-1, type=int)
parser.add_argument('-dl', '--downsample_level', default=16, type=int)
args = parser.parse_args()

image_name = args.image
patch_size = args.patch_size
stride = patch_size if args.stride == -1 else args.stride
downsample_lvl = args.downsample_level

file = h5py.File(f'dataset/images/{image_name}.hdf5')
reader = ImageReader(file[f'{downsample_lvl}x'], patch_size, stride)
print('Image ready, shape:', reader.shape[1], 'x', reader.shape[2])

model = m.unet()
path = args.model
model_name = path.split('/')[1]

# model.load_state_dict(torch.load(path))
model.load_state_dict(torch.load(path))

with torch.no_grad():
    pred = eval.make_prediction_patch_based(model, reader)

    cv2.imwrite(f'visuals/seg/prob-ps{patch_size}--{image_name}--{model_name}.png', (pred * 255).detach().cpu().numpy().astype(np.uint8))
    cv2.imwrite(f'visuals/seg/mask-ps{patch_size}--{image_name}--{model_name}.png', (eval.threshold(pred) * 255).detach().cpu().numpy().astype(np.uint8))
