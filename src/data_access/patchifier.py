import numpy as np

from .iter_mix_in import IterMixIn


class Patchifyer(IterMixIn):
    def __init__(self, height: int, width: int, patch_size: int, stride: int, overlap_last: bool = True):
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.stride = stride
        self.overlap_last = overlap_last

        n = int(np.ceil((height - patch_size) / stride)) + 1
        m = int(np.ceil((width - patch_size) / stride)) + 1
        self.shape = (n, m)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row = item[0]
            column = item[1]
        else:
            if item >= len(self):
                raise IndexError(f'index out of bounds: {item} of {len(self)}')
            row = int(np.floor(item / self.shape[1]))
            column = item % self.shape[1]
        return self.to_pixel(row, column)

    def __len__(self):
        return self.shape[0] * self.shape[1]

    def __iter__(self):
        self.current = 0

    def __next__(self):
        r, c = self[self.current]
        self.current += 1
        return r, c

    def to_pixel(self, row: int, column: int):
        if row >= self.shape[0]:
            raise IndexError(f'row out of bounds: {row} of {self.shape[0]}')
        elif column >= self.shape[1]:
            raise IndexError(f'row out of bounds: {column} of {self.shape[1]}')

        if row == self.shape[0] - 1 and self.overlap_last:
            r = self.height - self.patch_size
        else:
            r = row * self.stride
        if column == self.shape[1] - 1 and self.overlap_last:
            c = self.width - self.patch_size
        else:
            c = column * self.stride
        return r, c
