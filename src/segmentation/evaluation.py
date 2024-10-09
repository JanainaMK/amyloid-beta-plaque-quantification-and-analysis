import time
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_access import VsiReader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_prediction_patch_based(model: torch.nn.Module, image):
    if isinstance(image, VsiReader):
        loader = DataLoader(image, 1)
        model.to(DEVICE)
        model.eval()
        patch_shape = image.patch_it.shape
        result = torch.zeros((image.shape[0], image.shape[1])).to(DEVICE)
        row_var = 0
        time_var = time.time()
        for i, data in enumerate(loader):
            patch = data.to(DEVICE)
            patch = torch.permute(patch, (0, 3, 1, 2)).double()
            prediction = model(patch).detach()

            r = int(np.floor(i / patch_shape[1]))
            c = i % patch_shape[1]
            if r > row_var:
                print(f"row {r} done in {time.time() - time_var} seconds")
                row_var = r
                time_var = time.time()
            r_pixel = (
                image.shape[0] - image.patch_size
                if r == patch_shape[0] - 1
                else r * image.patch_size
            )
            c_pixel = (
                image.shape[1] - image.patch_size
                if c == patch_shape[1] - 1
                else c * image.patch_size
            )

            result[
                r_pixel : r_pixel + image.patch_size,
                c_pixel : c_pixel + image.patch_size,
            ] = prediction

        return result
    else:
        loader = DataLoader(image, 1)
        model.to(DEVICE)
        model.eval()
        patch_shape = image.patch_it.shape
        result = torch.zeros((image.shape[1], image.shape[2])).to(DEVICE)
        row_var = 0
        time_var = time.time()
        for i, data in enumerate(loader):
            patch = torch.Tensor(data).to(DEVICE)
            prediction = model(patch).detach()

            r = int(np.floor(i / patch_shape[1]))
            c = i % patch_shape[1]
            if r > row_var:
                print(f"row {r} done in {time.time() - time_var} seconds")
                row_var = r
                time_var = time.time()
            r_pixel = (
                image.shape[1] - image.patch_size
                if r == patch_shape[0] - 1
                else r * image.patch_size
            )
            c_pixel = (
                image.shape[2] - image.patch_size
                if c == patch_shape[1] - 1
                else c * image.patch_size
            )

            result[
                r_pixel : r_pixel + image.patch_size,
                c_pixel : c_pixel + image.patch_size,
            ] = prediction

        return result


def make_mask_patch_based(model: torch.nn.Module, image: VsiReader):
    prediction = make_prediction_patch_based(model, image)
    mask = threshold(prediction)
    return mask


def threshold(prediction, t=0.5):
    return (prediction > t).to(prediction.dtype)


def intersection_over_union(prediction: torch.Tensor, target: torch.Tensor):
    intersection = torch.sum(prediction * target)
    union = torch.sum((prediction + target) > 0.5)
    return intersection / union


def dice(prediction: torch.Tensor, target: torch.Tensor):
    intersection = torch.sum(prediction * target)
    denominator = torch.sum(prediction) + torch.sum(target)
    return 2 * intersection / denominator


Result = namedtuple("Result", ["index", "score", "image"])


def get_best(scores, images):
    i = np.argmax(scores)
    return Result(i, scores[i], images[i])


def get_median(scores, images):
    i = np.argsort(scores)[len(scores) // 2]
    return Result(i, scores[i], images[i])


def get_worst(scores, images):
    i = np.argmin(scores)
    return Result(i, scores[i], images[i])
