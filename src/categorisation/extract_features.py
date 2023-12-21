import torch
from torchvision.models import alexnet, AlexNet_Weights
import numpy as np
from sklearn.decomposition import PCA
from skimage.feature import hog

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ALEX = alexnet(AlexNet_Weights.DEFAULT).to(DEVICE)


def generate_alex_features(image_batch: torch.Tensor):
    if image_batch[0].shape != (3, 224, 224):
        raise ValueError('Input image should have shape (3, 224, 224), instead it is', image_batch[0].shape)
    with torch.no_grad():
        return ALEX(image_batch)


def pca(samples: np.ndarray, n_components: int):
    decomp = PCA(n_components=n_components, copy=True)
    return decomp.fit_transform(samples)


def generate_hog_features(image: np.ndarray):
    if image.shape != (256, 256):
        raise ValueError('Input image should have shape (256, 256), instead it is', image.shape)
    return hog(image, orientations=8, pixels_per_cell=(64, 64), cells_per_block=(2, 2))
