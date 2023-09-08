import numpy as np
import torch
from skimage.color import rgb2hed
from skimage.transform import resize

from src.util import from_skimg


def rgb_to_dab(image: np.ndarray):
    hed = rgb2hed(image)
    dab = hed[:, :, 2]
    return dab


def alex_prep(image: np.ndarray):
    if len(image.shape) == 2:
        image = np.tile(np.expand_dims(image, 2), (1, 1, 3))
    resized = resize(image, (224, 224))
    tensor = from_skimg(resized).to_torch()
    return torch.unsqueeze(tensor, 0)


def hog_prep(image: np.ndarray):
    resized = resize(image, (256, 256))
    return resized

