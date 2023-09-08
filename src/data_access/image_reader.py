import h5py
import numpy as np
import torch.utils.data as data

from . patchifier import Patchifyer
from .iter_mix_in import IterMixIn


class ImageReader(data.Dataset, IterMixIn):
    def __init__(self, dataset: h5py.Dataset, patch_size: int, stride: int, dtype=np.float64):
        self.data = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.dtype = dtype

        self.shape = self.data.shape
        self.patch_it = Patchifyer(self.shape[1], self.shape[2], self.patch_size, self.stride)

    def __len__(self):
        return len(self.patch_it)

    def __getitem__(self, item):
        r, c = self.patch_it[item]
        return self.data[:, r:r+self.patch_size, c:c+self.patch_size].astype(self.dtype)

    def get_full(self):
        return self.data[()].astype(self.dtype)

    def get_full_channel(self, channel):
        return self.data[channel].astype(self.dtype)

