import h5py
import numpy as np
import torch.utils.data as data

from .iter_mix_in import IterMixIn
from .patchifier import Patchifyer


class ImageReader(data.Dataset, IterMixIn):
    def __init__(
        self, dataset: h5py.Dataset, patch_size: int, stride: int, dtype=np.float64
    ):
        self.data = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.dtype = dtype
        self.shape = self.data.shape

        # Stores Patchifyer with generated patches from the image
        self.patch_it = Patchifyer(
            self.shape[1], self.shape[2], self.patch_size, self.stride
        )

    def __len__(self):
        # Returns the number of generated patches
        return len(self.patch_it)

    def __getitem__(self, item):
        # Retrieves a patch as an image at a given index
        row, col = self.patch_it[item]
        return self.data[
            :, row : row + self.patch_size, col : col + self.patch_size
        ].astype(self.dtype)

    def get_full(self):
        # Retrieves the full image data
        return self.data[()].astype(self.dtype)

    def get_full_channel(self, channel):
        # Retrieves the full image data for a specific channel
        return self.data[channel].astype(self.dtype)
