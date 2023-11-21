import time
import argparse
import sys

import torch
import numpy as np
import h5py
import bioformats

from src.segmentation.model import unet
from src.segmentation.evaluation import make_mask_patch_based
from src.data_access import VsiReader
import src.util.jvm as jvm
import src.util.cli as cli

start = time.time()

parser = argparse.ArgumentParser(prog='Segmentation', description='An algorithm that can segment grey matter from a vsi file')
parser.add_argument('image', type=str, help='The name of the vsi file that needs to be analysed. Omit .vsi in the name.')
cli.add_io_settings(parser)
cli.add_segmentation_settings(parser)


args = parser.parse_args()
print(args)

# file
image_name = args.image
vsi_root = args.source_folder
target_folder = args.target_folder

# parameters segmentation
patch_size_segmentation = args.patch_size_segmentation
downscale_factor = args.downscale_factor
model_path = args.model_path

if torch.cuda.is_available():
    print('cuda availible!')
    model = unet(model_path)
else:
    print('GPU requested, but CUDA is not available, exiting.')
    sys.exit()

result_file = h5py.File(f'{target_folder}/{image_name}.hdf5', 'a')

# this line sets the index of the full size image in the vsi file (see readme).
full_index = 13 if image_name[:5] == 'Image' else 0

jvm.start()
print('starting process', time.time() - start)
try:
    raw_reader = bioformats.ImageReader(f'{vsi_root}/{image_name}.vsi')
    vsi_reader = VsiReader(raw_reader, patch_size_segmentation, patch_size_segmentation, downscale_factor, np.uint8, False, True, full_index)
    print('segmentation reader loaded', time.time() - start)
    print('image shape:', vsi_reader.shape)
    print('patchifier shape:', vsi_reader.patch_it.shape)

    with torch.no_grad():
        grey_matter = make_mask_patch_based(model, vsi_reader).detach().cpu().numpy().astype(bool)
        if 'grey-matter' in result_file:
            del result_file['grey-matter']
        result_file.create_dataset('grey-matter', data=grey_matter)
        print('grey matter segmented', time.time() - start)
finally:
    jvm.stop()
    result_file.close()
