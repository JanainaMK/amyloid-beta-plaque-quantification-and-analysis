import h5py
import numpy as np
from torch.utils.data import Dataset

from .combined_reader import CombinedReader
from .iter_mix_in import IterMixIn

from src.util import LabelEnum


class DatasetReader(Dataset, IterMixIn):
    def __init__(self, data_group: h5py.Group, patch_size: int, stride: int, downsample_lvl: int, label_type: LabelEnum, dtype=np.float64):
        self.data_group = data_group
        self.patch_size = patch_size
        self.stride = stride
        self.downsample_lvl = downsample_lvl
        self.label_type = label_type
        self.dtype = dtype

        # Obtains a list of image names in the HDF5 data group
        self.image_list = [key for key in data_group.keys()]
        # Initializes a CombinedReader instance for each image
        self.readers = self.init_readers()
        # Number of images in the dataset
        self.num_images = len(self.image_list)
        # Calculates cumulative length of the dataset
        self.cumulative_length = self.calculate_cumulative_length()

    def init_readers(self):
        # Initializes a list of CombinedReader instances for each image in the dataset
        readers = []
        for image_name in self.image_list:
            readers.append(CombinedReader(
                self.data_group[image_name],
                self.patch_size,
                self.stride,
                self.downsample_lvl,
                self.label_type,
                self.dtype,
            ))
        return readers

    def calculate_cumulative_length(self):
        # Calculates the cumulative length of the dataset, based on the size of the CombinedReader instance for each image
        cumulative_length = []
        for i, reader in enumerate(self.readers):
            if i == 0:
                base = 0
            else:
                base = cumulative_length[-1]
            cumulative_length.append(base + len(reader))
        return cumulative_length

    def __getitem__(self, i):
        # Retrieves an item at a given global index.
        # Converts the global index to two indices: 
        # an image index pointing to an image in the dataset
        # a local index pointing to an item in the CombinedReader of the image
        image_index, local_index = self.convert_global_index(i)
        return self.readers[image_index][local_index]

    def __len__(self):
        # Returns the total number of items in the dataset
        return self.cumulative_length[-1]

    def convert_global_index(self, global_index: int):
        # Finds the appropriate image and local indices given a global index
        if global_index >= len(self):
            raise IndexError(f'index out of bounds: {global_index} of {len(self)}')
        return self.search(0, self.num_images - 1, global_index)

    def search(self, low: int, high: int, global_index: int):
        # Uses binary search to find image and local indices associated with a global index
        if high == low:
            image_index = low
            local_index = global_index - self.cumulative_length[low-1] if low != 0 else global_index
            return image_index, local_index
        mid = int(np.floor((high + low) / 2))
        if global_index < self.cumulative_length[mid]:
            return self.search(low, mid, global_index)
        else:
            return self.search(mid+1, high, global_index)

    def get_reader(self, i):
        # Retrieves a CombinedReader given a global index
        return self.readers[i]

    def get_image_and_label(self, i):
        # Retrieves image and label information from a CombinedReader
        return self.readers[i].image, self.readers[i].labels

    def get_num_images(self):
        # Returns the total number of images in the dataset
        return len(self.readers)
