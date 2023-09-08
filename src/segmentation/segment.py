import time

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from src.segmentation import PatchUnet, brainsec_resnet18
from src.data_access import ImageReader
from src.util import LabelEnum

start_time = time.time()

########
# CUDA #
########
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
print('cuda availible:', torch.cuda.is_available())

############
# SETTINGS #
############
image_name = 'Image_2014-037_Pgrijs_BA4'
model_name = '3d-experiment-resnet-smTrue-8x-bs32-ps256-e29'
downsample_lvl = 16
batch_size = 1
patch_size = 256
stride = 256
use_model = 'unet'
print(
    'Settings:',
    '\n\timage:', image_name,
    '\n\tmodel:', model_name,
    f'\n\tdownsample level: {downsample_lvl}x',
    '\n\tbatch size:', batch_size,
    '\n\tpatch size', patch_size,
    '\n\tstride:', stride,
)

###############
# PREPARATION #
###############

file = h5py.File(f'dataset/images/{image_name}.hdf5', 'r')
image_reader = ImageReader(file[f'{downsample_lvl}x'], patch_size, stride)
image_loader = DataLoader(image_reader, batch_size)
print(image_reader.shape)

if use_model == 'resnet':
    model = brainsec_resnet18().to(DEVICE)
elif use_model == 'unet':
    model = PatchUnet(3, 2)
else:
    raise ValueError(f'{use_model} is not a valid model to use')
model.load_state_dict(torch.load(f'models/{model_name}.pt'))
model.eval()

softmax = torch.nn.Softmax(1)

shape = image_reader.shape
seg = torch.zeros(shape[0], shape[1], dtype=torch.float).detach()
for i, data in enumerate(image_loader):
    r = int(np.floor(i*stride / shape[1]))*stride
    c = i*stride % shape[1]
    if c + patch_size > shape[1]:
        c = shape[1] - patch_size
    if r + patch_size > shape[0]:
        r = shape[0] - patch_size

    patches = torch.Tensor(data).to(DEVICE)
    out = softmax(model(patches)).detach().squeeze(0)
    seg[r:r+patch_size, c:c+patch_size] += out[1]
    if c + patch_size >= shape[1]:
        np.save(f'visuals/{model_name}-{image_name}.npy', seg.cpu().detach().numpy())
        print('row', r / stride, 'saved')






