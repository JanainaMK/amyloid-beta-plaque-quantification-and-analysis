import math


class PatchCoordinateIterator:
    """
    An interator that iterates over patches in a matrix-like structure. It returns the top left coordinates of each
    patch during iteration.
    """
    def __init__(
            self,
            width: int,
            height: int,
            patch_size: int,
            stride: int,
    ):
        """
        Constructor for the iterator
        :param width: Width of the matrix in pixels
        :param height: height of the matrix in pixels
        :param patch_size: Width and height of the patch
        :param stride:The stride of the sliding window
        """
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.stride = stride

        self.num_width = math.floor((width - patch_size) / stride) + 1
        self.num_height = math.floor((height - patch_size) / stride) + 1
        self.shape = self.num_height, self.num_width

        self.row = 0
        self.column = 0

    def __iter__(self):
        self.row = 0
        self.column = 0
        return self

    def __next__(self):
        """
        Get the next coordinates on the grid in a left-to-right then top-to-bottom order
        :return: The top left coordinates of the next patch.
        """
        row_pixel, column_pixel = self.to_pixel(self.row, self.column)

        self.column += 1
        if self.column > self.num_width:
            self.row += 1
            self.column = 0
        if self.row > self.num_height:
            raise StopIteration

        return row_pixel, column_pixel

    def to_pixel(self, row: int, column: int):
        if column < self.num_width:
            column_pixel = column * self.stride
        else:
            column_pixel = self.width - self.patch_size
        if row < self.num_height:
            row_pixel = row * self.stride
        else:
            row_pixel = self.height - self.patch_size
        return row_pixel, column_pixel

    def __getitem__(self, i):
        if i >= len(self) or i < 0:
            raise IndexError(f'index out of bounds: {i} of {len(self)}')
        row = math.floor(i/self.num_height)
        column = i % self.num_height
        return self.to_pixel(row, column)

    def __len__(self):
        return self.num_height * self.num_width





