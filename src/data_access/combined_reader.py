import h5py
import numpy as np
from torch.utils.data import Dataset

from src.util import LabelEnum

from .image_reader import ImageReader
from .iter_mix_in import IterMixIn
from .label_reader import LabelReader


# Reads Images combined with Labels
class CombinedReader(Dataset, IterMixIn):
    def __init__(
        self,
        image_group: h5py.Group,
        patch_size: int,
        stride: int,
        downsample_level: int,
        label_type: LabelEnum,
        dtype=np.float64,
    ):
        self.image_group = image_group
        self.patch_size = patch_size
        self.stride = stride
        self.downsample_level = downsample_level
        self.label_type = label_type
        self.dtype = dtype

        # Initializes LabelReader for reading labels
        self.labels = LabelReader(
            image_group[f"label_file/{downsample_level}x"],
            patch_size,
            stride,
            label_type,
            dtype,
        )
        # Initializes ImageReader for reading images
        self.image = ImageReader(
            image_group[f"image_file/{downsample_level}x"], patch_size, stride, dtype
        )

        # Checks if the shapes of images and labels match
        self.shape = None
        if self.image.shape[1:] == self.labels.shape:
            self.shape = self.labels.shape
        else:
            raise ValueError(
                f"image and label shapes do not match, image:{self.image.shape}, label:{self.labels.shape}"
            )

        # Stores the shape of the image patches
        self.patch_shape = self.image.patch_it.shape

    def __getitem__(self, i):
        # Retrieves an item (image, label) at a given index
        if i >= len(self):
            raise IndexError(f"local index out of bounds. {i} of {len(self)}")
        return self.image[i], self.labels[i]

    def __len__(self):
        # Returns the length of the dataset
        image_len = len(self.image)
        label_len = len(self.labels)
        if image_len != label_len:
            raise ValueError(
                f"Patch iterators are of different lengths: image {image_len}, labels {label_len}"
            )
        return image_len
