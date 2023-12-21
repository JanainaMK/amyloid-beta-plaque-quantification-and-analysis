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
        
        # Initialize Patchifyer to generate patches
        self.patch_it = Patchifyer(height, width, self.patch_size, self.stride)

    def to_label(self, patch):
        # Convert a patch to a label based on the specified label type.
        if self.label_type == LabelEnum.PIXEL:
            # If using pixel-wise labeling, return the entire patch
            return patch
        elif self.label_type == LabelEnum.CLASS_MIDDLE:
            # If using middle pixel labeling, return the value at the middle pixel
            i = int(np.floor(self.patch_size / 2))
            return patch[i, i]
        elif self.label_type == LabelEnum.CLASS_AVG:
            # If using average pixel labeling, return the mean value of the patch
            return patch.mean()
        else:
            return None

    def __getitem__(self, i):
        if self.label_type == LabelEnum.NO:
            # If no labeling is specified, return None
            return None
        # Get top-left coordinates of the patch
        top, left = self.patch_it[i]
        # Extract the patch from the label data
        patch = self.data[top:top + self.patch_size, left:left + self.patch_size]
        patch = np.expand_dims(patch, 0).astype(self.dtype)
        # Convert the patch to a labeled value based on the label type
        return self.to_label(patch)

    def __len__(self):
        #  Get the total number of patches.
        return len(self.patch_it)

    def get_full(self):
        # Get the full label data.
        return self.data[()].astype(self.dtype)
