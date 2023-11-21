import time
import argparse
import sys

import torch
import numpy as np
import h5py
import bioformats
from skimage.transform import resize

from src.segmentation.model import unet
from src.segmentation.evaluation import make_mask_patch_based
import src.localisation.plaque as p
import src.localisation.plaque_find as pf
import src.localisation.whole_slide_analysis as wsa
from src.data_access import VsiReader
import src.util.jvm as jvm

start = time.time()

print('localisation started')
parser = argparse.ArgumentParser(prog='Segmentation and Localisation pipeline', description='An algorithm that can crop amyloid beta plaques from a vsi file')
parser.add_argument('image', type=str, help='The name of the vsi file that needs to be analysed. Omit .vsi in the name.')
parser.add_argument('-sf', '--source_folder', required=True, type=str, help='the folder in which the vsi file is located')
parser.add_argument('-tf', '--target_folder', default='result/images', type=str, help='The folder in which to place the resulting hdf5 file.')
parser.add_argument('-psl', '--patch_size_localisation', default=4096, type=int, help='The patch size in pixels used in the localisation step (default=4096)')
parser.add_argument('-pss', '--patch_size_segmentation', default=1024, type=int, help='The patch size in pixels used in the segmentation step (default=1024)')
parser.add_argument('-t', '--threshold', default=0.04, type=float, help='The minimum threshold parameter for the localisation step. Range from 0 to 1 (default=0.04)')
parser.add_argument('-ks', '--kernel_size', default=21, type=int, help='The kernel size parameter in pixels for the localisation step (default=21)')
parser.add_argument('-ms', '--minimum_size', default=10, type=float, help='The minimum plaque size parameter in microns for the localisation step (default=10)')
parser.add_argument('-df', '--downscale_factor', default=16, type=int, help='The downsampling factor used for creating the grey matter segmentation (default=16)')
parser.add_argument('-ss', '--segmentation_setting', choices=['no', 'load', 'create'], default='create', help='The way the grey matter segmentation is handled. \n \'no\' skips the segmentation \n \'load\' loads the segmentation from the result file located in the target_folder \n \'create\' creates a new segmentation and stores it in the target_folder.')


args = parser.parse_args()
print(args)

# file
image_name = args.image
vsi_root = args.source_folder
target_folder = args.target_folder

# parameters segmentation
patch_size_segmentation = args.patch_size_segmentation
downscale_factor = args.downscale_factor
segmentation_setting = args.segmentation_setting

# parameters localisation
patch_size_localisation = args.patch_size_localisation
threshold = int(args.threshold * 255)
kernel_size = args.kernel_size
min_size_micron = args.minimum_size
minsize = int(min_size_micron / 0.274)


cuda_availible = torch.cuda.is_available()
if segmentation_setting == 'create' and cuda_availible:
    print('cuda:', cuda_availible)
    model = unet('models/2023-03-15-unet-16x-bs16-ps256-lr0.0001/2023-03-15-unet-16x-bs16-ps256-lr0.0001-e3v49.pt')
elif segmentation_setting == 'create' and not cuda_availible:
    print('GPU requested, but CUDA is not available, exiting.')
    sys.exit()
else:
    model = None

result_file = h5py.File(f'{target_folder}/{image_name}.hdf5', 'a')
if 'plaques' in result_file:
    del result_file['plaques']
    print('old plaques deleted')
plaque_group = result_file.create_group('plaques')

# this line sets the index of the full size image in the vsi file (see readme).
full_index = 13 if image_name[:5] == 'Image' else 0

jvm.start()
print('starting process', time.time() - start)
try:
    raw_reader = bioformats.ImageReader(f'{vsi_root}/{image_name}.vsi')

    # segmentation
    vsi_reader = VsiReader(raw_reader, patch_size_segmentation, patch_size_segmentation, downscale_factor, np.uint8, False, True, full_index)
    print('segmentation reader loaded', time.time() - start)
    print('image shape:', vsi_reader.shape)
    print('patchifier shape:', vsi_reader.patch_it.shape)

    if segmentation_setting == 'no':
        grey_matter = None
    elif segmentation_setting == 'create':
        with torch.no_grad():
            grey_matter = make_mask_patch_based(model, vsi_reader).detach().cpu().numpy().astype(bool)
            if 'grey-matter' in result_file:
                del result_file['grey-matter']
            result_file.create_dataset('grey-matter', data=grey_matter)
            print('grey matter segmented', time.time() - start)
    elif 'grey-matter' not in result_file:
        raise FileNotFoundError(f'there is no gray matter found for {image_name}')
    else:
        grey_matter = result_file['grey-matter'][()]
        print('grey matter loaded', time.time() - start)
    print()

    # localisation
    vsi_reader = VsiReader(raw_reader, patch_size_localisation, patch_size_localisation, 0, np.uint8, False, False, full_index)
    print('localisation reader loaded', time.time() - start)
    print('image shape:', vsi_reader.shape)
    print('patchifier shape:', vsi_reader.patch_it.shape)

    pms = []

    for i in range(len(vsi_reader)):
        patch = vsi_reader[i]
        r, c = vsi_reader.patch_it[i]
        h = min(vsi_reader.shape[0] - r, vsi_reader.patch_size)
        w = min(vsi_reader.shape[1] - c, vsi_reader.patch_size)
        print('analysing patch', f'{i}/{len(vsi_reader)}, loc:({r}, {c}), shape: ({h}, {w})')

        gm = wsa.get_grey_matter(r, c, h, w, grey_matter, downscale_factor)
        if gm.mean() < 0.01:
            print('no grey matter', time.time() - start)
            print()
            continue
        gm = resize(gm, (h, w))
        print('grey matter found', time.time() - start)

        binary, t = pf.dab_threshold_otsu(patch, threshold)
        binary = gm * binary
        closed = pf.closing(binary, kernel_size)
        plaques, mask = pf.find_plaques(closed, minsize, True, True)
        print(len(plaques), 'plaques found', time.time() - start)
        print('threshold used:', t)
        print()

        for pm in plaques:
            x = c + pm.bb.x
            y = r + pm.bb.y
            final_pm = p.PlaqueMask(pm.mask, p.BoundingBox(x, y, pm.bb.w, pm.bb.h))
            pms.append(final_pm)

    print('slide analysed in', time.time() - start, 'seconds')
    print()
    print('initial detections:', len(pms))
    print('merging...')
    pairs = wsa.get_bordering_detections_from_plaque_mask(pms)
    print(len(pairs), 'bordering detections',  time.time() - start)
    ccs = wsa.get_connected_components(pairs, len(pms))
    print('detections after merging:', len(ccs), time.time() - start)
    wsa.merge_plaque_masks_from_list(pms, ccs)
    print('detections merged')

    # result
    print('saving detections...')
    n = len(pms)
    plaque_group.attrs['length'] = n
    bbs = np.zeros((n, 4), dtype=int)
    areas = np.zeros(n)
    roundnesses = np.zeros(n)
    for i, pm in enumerate(pms):
        plaque_img = vsi_reader.image_reader.rdr.openBytesXYWH(0, pm.bb.x, pm.bb.y, pm.bb.w, pm.bb.h)
        plaque_img = plaque_img.reshape((pm.bb.h, pm.bb.w, 3))

        bbs[i] = [pm.bb.x, pm.bb.y, pm.bb.w, pm.bb.h]
        areas[i] = np.sum(pm.mask) * 0.274**2  # micron^2
        roundnesses[i] = p.roundness(pm.mask)

        plaque_entry = plaque_group.create_group(str(i))
        plaque_entry.create_dataset('plaque', data=plaque_img)
        plaque_entry.create_dataset('mask', data=pm.mask)
        plaque_entry.attrs['x'] = pm.bb.x
        plaque_entry.attrs['y'] = pm.bb.y
        plaque_entry.attrs['w'] = pm.bb.w
        plaque_entry.attrs['h'] = pm.bb.h
        plaque_entry.attrs['area'] = areas[i]
        plaque_entry.attrs['roundness'] = roundnesses[i]

    if 'bbs' in result_file:
        del result_file['bbs']
    if 'area' in result_file:
        del result_file['area']
    if 'roundness' in result_file:
        del result_file['roundness']
    result_file.create_dataset('bbs', data=bbs)
    result_file.create_dataset('area', data=areas)
    result_file.create_dataset('roundness', data=roundnesses)

    gm = result_file['grey-matter'][()]
    gm_area = np.sum(gm) * (0.274 * 16) ** 2  # micron^2
    gm_area = gm_area / 1000000  # mm^2

    result_file.attrs['n_plaques'] = n
    result_file.attrs['ppmm'] = n / gm_area

    print('plaques saved')


finally:
    jvm.stop()
    result_file.close()



