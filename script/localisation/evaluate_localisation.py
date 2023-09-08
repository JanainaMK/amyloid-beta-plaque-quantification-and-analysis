import argparse
import time
import random
from pathlib import Path

import h5py
from skimage.io import imsave

import src.localisation.plaque_find as pf
import src.localisation.evaluation as ev
import src.localisation.visualisation as vis
from src.util import ConfusionMatrix

parser = argparse.ArgumentParser(description='Localisation evaluation')

parser.add_argument('-dt', '--dab_threshold', default=0.1, type=float)
parser.add_argument('-ks', '--kernel_size', default=21, type=int)
parser.add_argument('-ms', '--minimum_size', default=10, type=float)
parser.add_argument('-pc', '--positive_criterion', default=0.5, type=float)
parser.add_argument('-dto', '--dab_threshold_otsu', action='store_true')
parser.add_argument('-v', '--visualise', action='store_true')
parser.add_argument('-m', '--use_mask', action='store_true')
parser.add_argument('-s', '--strict', action='store_true')
parser.add_argument('-r', '--rescale', action='store_true')
args = parser.parse_args()
print(args)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

dab_threshold_rel = args.dab_threshold
dab_threshold = int(dab_threshold_rel * 255)
kernel_size = args.kernel_size
min_size_micron = args.minimum_size
minsize = int(min_size_micron / 0.274)
positive_criterion = args.positive_criterion
use_otsu = args.dab_threshold_otsu
visualise = args.visualise
use_mask = args.use_mask
strict = args.strict
rescale = args.rescale

dab_threshold_string = f'o{dab_threshold_rel}' if use_otsu else dab_threshold_rel
param_string = f'dt{dab_threshold_string}-ks{kernel_size}-ms{min_size_micron}'

file = h5py.File('dataset/localisation.hdf5')

if visualise:
    Path(f'visuals/bbs/{param_string}').mkdir(exist_ok=True)


total_matrix = ConfusionMatrix()
for key in file:
    start = time.time()
    print('evaluating:', key)
    image_entry = file[key]
    image = image_entry['image'][()]
    thresh, t = pf.dab_threshold_otsu(image, dab_threshold, rescale) if use_otsu else pf.dab_threshold(image, dab_threshold, rescale)
    closed = pf.closing(thresh, kernel_size)
    predictions, full_prediction_mask = pf.find_plaques(closed, minsize, use_mask)
    ground_truth = ev.get_labels(image_entry['annotations'], use_mask)
    if use_mask:
        confusion_matrix = ev.evaluate_masks(predictions, ground_truth, positive_criterion, strict)
    else:
        confusion_matrix = ev.evaluate_bounding_boxes(predictions, ground_truth, positive_criterion, strict)
    total_matrix += confusion_matrix
    print('threshold value:', t)
    print('predictions:', len(predictions))
    print('actual:', len(ground_truth))
    print(confusion_matrix)
    print('precision:', confusion_matrix.precision())
    print('recall:', confusion_matrix.recall())
    print()
    if visualise:
        if use_mask:
            drawn_predictions = vis.draw_full_mask(image, full_prediction_mask, BLUE)
            drawn = vis.draw_masks(drawn_predictions, ground_truth, GREEN)
        else:
            drawn_predictions = vis.draw_bounding_boxes(image, predictions, BLUE)
            drawn = vis.draw_bounding_boxes(drawn_predictions, ground_truth, GREEN)
        imsave(f'visuals/bbs/{param_string}/{param_string}-{key}.png', drawn)

print()
print('number of predictions:', total_matrix.TP + total_matrix.FP)
print('actual number of plaques:', total_matrix.TP + total_matrix.FN)
print('precision:', total_matrix.precision())
print('recall:', total_matrix.recall())

ev.write_evaluation_to_hdf5(param_string, total_matrix)
