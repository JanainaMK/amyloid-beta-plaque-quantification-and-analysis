import cv2
import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Matrix:
    def __init__(self, matrix: np.ndarray):
        if len(matrix.shape) != 3:
            raise ValueError(f'input must be a color matrix. current shape {matrix.shape}')
        if matrix.shape[2] != 3:
            raise ValueError(f'color matrix should have shape (h, w, 3). current shape {matrix.shape}')
        self.matrix = matrix

    def to_skimg(self):
        return self.matrix

    def to_cv(self):
        return cv2.cvtColor(self.matrix, cv2.COLOR_RGB2BGR)

    def to_np(self):
        return np.transpose(self.matrix, (2, 0, 1))

    def to_torch(self):
        temp = self.to_np()
        return torch.Tensor(temp).to(DEVICE)

    def to_png(self, target_file):
        cv2.imwrite(target_file, self.to_cv())


def from_skimg(matrix: np.ndarray) -> Matrix:
    return Matrix(matrix)


def from_cv(matrix: np.ndarray) -> Matrix:
    temp = cv2.cvtColor(matrix, cv2.COLOR_BGR2RGB)
    return Matrix(temp)


def from_np(matrix: np.ndarray) -> Matrix:
    temp = np.transpose(matrix, (1, 2, 0))
    return Matrix(temp)


def from_torch(matrix: torch.Tensor) -> Matrix:
    temp = matrix.detach().cpu().numpy()
    return from_np(temp)


def is_np(image):
    return image.shape[0] == 3


def is_cv(image):
    return image.shape[2] == 3
