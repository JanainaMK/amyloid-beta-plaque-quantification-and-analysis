import h5py
import numpy as np
from torch.utils.data import Dataset

from .patchifier import Patchifyer
from .iter_mix_in import IterMixIn
from src.util import LabelEnum


class LabelReader(Dataset, IterMixIn):
    dtype = np.float64

    def __init__(self, data: h5py.Dataset, patch_size: int, stride: int, label_type: LabelEnum, dtype=np.float64):
        self.data = data
        self.patch_size = patch_size
        self.stride = stride
        self.label_type = label_type
        self.dtype = dtype

        height, width = self.data.shape
        self.shape = self.data.shape
        self.patch_it = Patchifyer(height, width, self.patch_size, self.stride)

    def to_label(self, patch):
        if self.label_type == LabelEnum.PIXEL:
            return patch
        elif self.label_type == LabelEnum.CLASS_MIDDLE:
            i = int(np.floor(self.patch_size / 2))
            return patch[i, i]
        elif self.label_type == LabelEnum.CLASS_AVG:
            return patch.mean()
        else:
            return None

    def __getitem__(self, i):
        if self.label_type == LabelEnum.NO:
            return None
        top, left = self.patch_it[i]
        patch = self.data[top:top + self.patch_size, left:left + self.patch_size]
        patch = np.expand_dims(patch, 0).astype(self.dtype)
        return self.to_label(patch)

    def __len__(self):
        return len(self.patch_it)

    def get_full(self):
        return self.data[()].astype(self.dtype)
