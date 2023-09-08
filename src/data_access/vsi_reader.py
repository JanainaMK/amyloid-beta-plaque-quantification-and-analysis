import math

import javabridge
import bioformats
import torch.utils.data as data
import numpy as np

from .patchifier import Patchifyer
from .iter_mix_in import IterMixIn


class VsiReader(data.Dataset, IterMixIn):

    FULL_INDEX = 13
    SIZE_FULL_LOAD = 4096

    def __init__(
            self,
            image_reader: bioformats.ImageReader,
            patch_size: int,
            stride: int,
            downsample_lvl: int,
            dtype=np.float64,
            preload=False,
            overlap_last: bool = True,
            case: str = 'centenarian',
    ):
        self.image_reader = image_reader
        self.patch_size = patch_size
        self.stride = stride
        self.downsample_lvl = downsample_lvl
        self.dtype = dtype
        self.preload = preload
        self.overlap_last = overlap_last

        self.full_index = 0 if case == 'AD' else 13
        self.fully_loaded = False

        lvl = np.log2(downsample_lvl) if downsample_lvl != 0 else 0
        self.image_reader.rdr.setSeries(self.full_index + lvl)
        self.shape = self.image_reader.rdr.getSizeY(), image_reader.rdr.getSizeX()
        self.patch_it = Patchifyer(self.shape[0], self.shape[1], self.patch_size, self.stride, self.overlap_last)

        self.image = None
        if preload:
            self.image = self.get_full()

    def __len__(self):
        return len(self.patch_it)

    def __getitem__(self, item):
        r, c = self.patch_it[item]
        if self.fully_loaded:
            return self.image[:, r:r+self.patch_size, c:c+self.patch_size]
        else:
            w = min(self.shape[1] - c, self.patch_size)
            h = min(self.shape[0] - r, self.patch_size)
            patch = self.image_reader.rdr.openBytesXYWH(0, c, r, w, h)
            return patch.reshape((h, w, 3)).astype(self.dtype)

    def get_full(self):
        if not self.fully_loaded:
            self.image = np.zeros((self.shape[0], self.shape[1], 3)).astype(self.dtype)
            temp_patch_it = Patchifyer(self.shape[0], self.shape[1], self.SIZE_FULL_LOAD, self.SIZE_FULL_LOAD)
            for i in range(len(temp_patch_it)):
                r, c = temp_patch_it[i]
                patch = self.image_reader.rdr.openBytesXYWH(0, c, r, self.SIZE_FULL_LOAD, self.SIZE_FULL_LOAD)
                patch = patch.reshape((self.SIZE_FULL_LOAD, self.SIZE_FULL_LOAD, 3)).astype(self.dtype)
                self.image[r:r+self.SIZE_FULL_LOAD, c:c+self.SIZE_FULL_LOAD] = patch
            self.fully_loaded = True
        return self.image

    def get_full_channel(self, channel):
        return self.get_full()[:, :, channel]



