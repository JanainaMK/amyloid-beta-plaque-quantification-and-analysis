import time

import h5py
import numpy as np

from src.localisation import BoundingBox, PlaqueMask
from src.util import ConfusionMatrix


def get_labels(annotations: h5py.Group, return_masks):
    labels = [None] * len(annotations)
    for i, name in enumerate(annotations):
        bb = BoundingBox(
            annotations[f'{name}/x'][()],
            annotations[f'{name}/y'][()],
            annotations[f'{name}/w'][()],
            annotations[f'{name}/h'][()],
        )
        if return_masks:
            mask = annotations[f'{name}/shape'][()]
            labels[i] = PlaqueMask(mask, bb)
        else:
            labels[i] = bb
    return labels


def calculate_bounding_box_iou(bb0: BoundingBox, bb1: BoundingBox):
    x_overlap = max(0, min(bb0.x + bb0.w, bb1.x + bb1.w) - max(bb0.x, bb1.x))
    y_overlap = max(0, min(bb0.y + bb0.w, bb1.y + bb1.w) - max(bb0.y, bb1.y))
    intersection = x_overlap * y_overlap
    union = bb0.h * bb0.w + bb1.h * bb1.w - intersection
    return intersection / union


def calculate_mask_iou(pm0: PlaqueMask, pm1: PlaqueMask):
    width = max(pm0.bb.x + pm0.bb.w, pm1.bb.x + pm1.bb.w) - min(pm0.bb.x, pm1.bb.x)
    height = max(pm0.bb.y + pm0.bb.h, pm1.bb.y + pm1.bb.h) - min(pm0.bb.y, pm1.bb.y)
    if width >= pm0.bb.w + pm1.bb.w or height >= pm0.bb.h + pm1.bb.h:
        return 0
    overlap = np.zeros((height, width), dtype=np.uint8)
    overlap = paste_plaque_mask(overlap, pm0)
    overlap = paste_plaque_mask(overlap, pm1)
    intersection = np.sum(overlap == 2)
    union = np.sum(overlap > 0)
    if union == 0:
        return 0
    else:
        return intersection / union


def calculate_mask_iop(prediction: PlaqueMask, label: PlaqueMask):
    width = max(prediction.bb.x + prediction.bb.w, label.bb.x + label.bb.w) - min(prediction.bb.x, label.bb.x)
    height = max(prediction.bb.y + prediction.bb.h, label.bb.y + label.bb.h) - min(prediction.bb.y, label.bb.y)
    if width >= prediction.bb.w + label.bb.w or height >= prediction.bb.h + label.bb.h:
        return 0
    overlap = np.zeros((height, width), dtype=np.uint8)
    overlap = paste_plaque_mask(overlap, prediction)
    overlap = paste_plaque_mask(overlap, label)
    intersection = np.sum(overlap == 2)
    return intersection / np.sum(prediction.mask)


def paste_plaque_mask(target: np.ndarray, pm: PlaqueMask):
    x0 = min(pm.bb.x, pm.bb.x) - pm.bb.x
    y0 = min(pm.bb.y, pm.bb.y) - pm.bb.y
    target[y0:y0+pm.bb.h, x0:x0+pm.bb.w] += pm.mask.astype(np.uint8)
    return target


def make_performance_matrix(predictions, ground_truth, func):
    if len(predictions) == 0:
        raise ValueError('There are no predictions to make the performance matrix')
    if len(ground_truth) == 0:
        raise ValueError('There is no ground truth to make the performance matrix')
    matrix = np.zeros((len(predictions), len(ground_truth)))
    for i, pred in enumerate(predictions):
        for j, label in enumerate(ground_truth):
            matrix[i, j] = func(pred, label)
    return matrix


def make_bounding_box_iou_matrix(predictions, ground_truth):
    return make_performance_matrix(predictions, ground_truth, calculate_bounding_box_iou)


def make_mask_iou_matrix(predictions, ground_truth):
    return make_performance_matrix(predictions, ground_truth, calculate_mask_iou)


def make_mask_iop_matrix(predictions, ground_truth):
    return make_performance_matrix(predictions, ground_truth, calculate_mask_iop)


def evaluate_performance_matrix_strict(matrix: np.ndarray, threshold):
    confusion_matrix = ConfusionMatrix()
    result = np.argmax(matrix, axis=1)
    for i, p in enumerate(result):
        if matrix[i, p] < threshold:
            confusion_matrix.FP += 1
        elif np.argmax(matrix[:, p]) == i:
            confusion_matrix.TP += 1
    confusion_matrix.FN = matrix.shape[1] - confusion_matrix.TP
    return confusion_matrix


def evaluate_performance_matrix_loose(matrix: np.ndarray, threshold):
    confusion_matrix = ConfusionMatrix()
    plaque_found = np.zeros(matrix.shape[1], dtype=bool)
    result = np.argmax(matrix, axis=1)
    for i, p in enumerate(result):
        if matrix[i, p] < threshold:
            confusion_matrix.FP += 1
        else:
            confusion_matrix.TP += 1
            plaque_found[p] = True
    confusion_matrix.FN = len(plaque_found) - np.sum(plaque_found)
    return confusion_matrix


def evaluate(predictions, ground_truth, threshold, strict, matrix_func):
    if len(predictions) == 0:
        return ConfusionMatrix(FN=len(ground_truth))
    if len(ground_truth) == 0:
        return ConfusionMatrix(FP=len(predictions))
    matrix = matrix_func(predictions, ground_truth)
    if strict:
        return evaluate_performance_matrix_strict(matrix, threshold)
    else:
        return evaluate_performance_matrix_loose(matrix, threshold)


def evaluate_bounding_boxes(predictions, ground_truth, threshold, strict):
    return evaluate(predictions, ground_truth, threshold, strict, make_bounding_box_iou_matrix)


def evaluate_masks(predictions, ground_truth, threshold, strict):
    return evaluate(predictions, ground_truth, threshold, strict, make_mask_iop_matrix)


EVAL_FILE = 'evaluation/localisation-proper.hdf5'


def write_evaluation_to_hdf5(param_string: str, confusion_matrix: ConfusionMatrix):
    while True:
        try:
            file = h5py.File(EVAL_FILE, 'a')
            group = file.create_group(param_string)
            group.attrs['TP'] = confusion_matrix.TP
            group.attrs['FP'] = confusion_matrix.FP
            group.attrs['FN'] = confusion_matrix.FN
            group.attrs['TN'] = confusion_matrix.TN
            group.attrs['precision'] = confusion_matrix.precision()
            group.attrs['recall'] = confusion_matrix.recall()
            file.close()
            break
        except OSError:
            time.sleep(3)


def print_eval():
    file = h5py.File(EVAL_FILE, 'r')
    for key in file:
        p = f'{key}/precision'
        r = f'{key}/recall'
        print(key.rjust(20), f'avg precision:{file[p][()]:.3f}\tavg recall:{file[r][()]:.3f}')
    file.close()











