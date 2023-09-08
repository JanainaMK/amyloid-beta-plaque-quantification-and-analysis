import h5py
import numpy as np
from torch.utils.data import Dataset

from .image_reader import ImageReader
from .label_reader import LabelReader
from .iter_mix_in import IterMixIn
from src.util import LabelEnum


class CombinedReader(Dataset, IterMixIn):
    def __init__(self, image_group: h5py.Group, patch_size: int, stride: int, downsample_level: int, label_type: LabelEnum, dtype=np.float64):
        self.image_group = image_group
        self.patch_size = patch_size
        self.stride = stride
        self.downsample_level = downsample_level
        self.label_type = label_type
        self.dtype = dtype

        self.labels = LabelReader(image_group[f'label_file/{downsample_level}x'], patch_size, stride, label_type, dtype)
        self.image = ImageReader(image_group[f'image_file/{downsample_level}x'], patch_size, stride, dtype)

        self.shape = None
        if self.image.shape[1:] == self.labels.shape:
            self.shape = self.labels.shape
        else:
            raise ValueError(f'image and label shapes do not match, image:{self.image.shape}, label:{self.labels.shape}')

        self.patch_shape = self.image.patch_it.shape

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError(f'local index out of bounds. {i} of {len(self)}')
        return self.image[i], self.labels[i]

    def __len__(self):
        image_len = len(self.image)
        label_len = len(self.labels)
        if image_len != label_len:
            raise ValueError(f"Patch iterators are of different lengths: image {image_len}, labels {label_len}")
        return image_len
