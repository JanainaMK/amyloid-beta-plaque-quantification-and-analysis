import time
import argparse
import sys

import torch
import numpy as np
import h5py
import bioformats
from skimage.transform import resize

print(sys.path)

from src.segmentation.model import unet
from src.segmentation.evaluation import make_mask_patch_based
import src.localisation.plaque as p
import src.localisation.plaque_find as pf
import src.localisation.whole_slide_analysis as wsa
from src.data_access import VsiReader
import src.util.jvm as jvm

print('localisation started')
parser = argparse.ArgumentParser()
parser.add_argument('image', type=str)
parser.add_argument('-psl', '--patch_size_localisation', default=4096, type=int)
parser.add_argument('-pss', '--patch_size_segmentation', default=1024, type=int)
parser.add_argument('-t', '--threshold', default=0.04, type=float)
parser.add_argument('-ks', '--kernel_size', default=21, type=int)
parser.add_argument('-ms', '--minimum_size', default=10, type=float)
parser.add_argument('-df', '--downscale_factor', default=16, type=int)
parser.add_argument('-tf', '--target_folder', default='result/images', type=str)
parser.add_argument('-uo', '--use_otsu', action='store_true')
parser.add_argument('-ss', '--skip_segmentation', action='store_true')
parser.add_argument('-gpu', '--use_gpu', action='store_true')
parser.add_argument('-c', '--case', default='', type=str)


args = parser.parse_args()
print(args)

image_name = args.image
patch_size_localisation = args.patch_size_localisation
patch_size_segmentation = args.patch_size_segmentation
threshold = int(args.threshold * 255)
use_otsu = args.use_otsu
kernel_size = args.kernel_size
min_size_micron = args.minimum_size
minsize = int(min_size_micron / 0.274)
downscale_factor = args.downscale_factor
skip_segmentation = args.skip_segmentation
use_gpu = args.use_gpu
case = args.case


start = time.time()
cuda_availible = torch.cuda.is_available()
if use_gpu and cuda_availible:
    print('cuda:', cuda_availible)
    model = unet('models/2023-03-15-unet-16x-bs16-ps256-lr0.0001/2023-03-15-unet-16x-bs16-ps256-lr0.0001-e3v49.pt')
elif use_gpu and not cuda_availible:
    print('GPU requested, but CUDA is not availible, exiting.')
    sys.exit()
else:
    model = None

# file_16x = h5py.File('dataset/AD+cent.hdf5', 'r')
result_file = h5py.File(f'{args.target_folder}/{image_name}.hdf5', 'a')
if 'plaques' in result_file:
    del result_file['plaques']
    print('old plaques deleted')
plaque_group = result_file.create_group('plaques')

print(case, 'case')
if 'AD' in case:
    vsi_root = 'dataset/Abeta images AD cohort'
elif case == 'centenarian':
    vsi_root = 'dataset/Abeta images 100+'
else:
    print('case not specified: exiting.')
    sys.exit()

jvm.start()
print('starting process', time.time() - start)
try:
    raw_reader_0x = bioformats.ImageReader(f'{vsi_root}/{image_name}.vsi')
    raw_reader_16x = bioformats.ImageReader(f'{vsi_root}/{image_name}.vsi')
    vsi_reader_0x = VsiReader(raw_reader_0x, patch_size_localisation, patch_size_localisation, 0, np.uint8, False, False, case)
    vsi_reader_16x = VsiReader(raw_reader_16x, patch_size_segmentation, patch_size_segmentation, downscale_factor, np.uint8, False, True, case)
    print('readers loaded', time.time() - start)
    print('0x image shape:', vsi_reader_0x.shape)
    print('16x image shape:', vsi_reader_16x.shape)
    print('0x patchifier shape:', vsi_reader_0x.patch_it.shape)
    print('16x patchifier shape:', vsi_reader_16x.patch_it.shape)

    if skip_segmentation:
        grey_matter = None
    elif use_gpu:
        with torch.no_grad():
            grey_matter = make_mask_patch_based(model, vsi_reader_16x).detach().cpu().numpy().astype(bool)
            if 'grey-matter' in result_file:
                del result_file['grey-matter']
            result_file.create_dataset('grey-matter', data=grey_matter)
            print('grey matter segmented', time.time() - start)
    elif 'grey-matter' not in result_file:
        raise FileNotFoundError(f'there is no gray matter found for {image_name}')
    else:
        grey_matter = result_file['grey-matter'][()]
        print('grey matter found', time.time() - start)

    pms = []

    for i in range(len(vsi_reader_0x)):
        patch = vsi_reader_0x[i]
        r, c = vsi_reader_0x.patch_it[i]
        h = min(vsi_reader_0x.shape[0] - r, vsi_reader_0x.patch_size)
        w = min(vsi_reader_0x.shape[1] - c, vsi_reader_0x.patch_size)
        print('analysing patch', f'{i}/{len(vsi_reader_0x)}, loc:({r}, {c}), shape: ({h}, {w}), alt shape:', patch.shape)

        gm = wsa.get_grey_matter(r, c, h, w, grey_matter, downscale_factor)
        if gm.mean() < 0.01:
            print('no grey matter', time.time() - start)
            print()
            continue
        gm = resize(gm, (h, w))
        print('grey matter found', time.time() - start)

        binary, t = pf.dab_threshold_otsu(patch, threshold) if use_otsu else pf.dab_threshold(patch, threshold)
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

    print('saving detections...')
    n = len(pms)
    plaque_group.attrs['length'] = n
    bbs = np.zeros((n, 4), dtype=int)
    areas = np.zeros(n)
    roundnesses = np.zeros(n)
    for i, pm in enumerate(pms):
        plaque_img = vsi_reader_0x.image_reader.rdr.openBytesXYWH(0, pm.bb.x, pm.bb.y, pm.bb.w, pm.bb.h)
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
    # file_16x.close()
    result_file.close()



