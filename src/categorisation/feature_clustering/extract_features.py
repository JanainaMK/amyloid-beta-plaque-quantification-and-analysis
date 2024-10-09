import numpy as np
import torch
import torch.nn as nn
from skimage.feature import hog
from sklearn.decomposition import PCA
from torchvision.models import AlexNet_Weights, ResNet18_Weights, alexnet, resnet18

from src.categorisation.autoencoders.vae import VAE


def generate_features(image_batch: torch.Tensor, feature_src: str):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if image_batch[0].shape != (3, 224, 224):
        raise ValueError(
            "Input image should have shape (3, 224, 224), instead it is",
            image_batch[0].shape,
        )
    with torch.no_grad():
        if feature_src == "alex":
            alex_model = alexnet(AlexNet_Weights.DEFAULT).to(DEVICE)
            alex_model.eval()
            return alex_model(image_batch)
        elif "res":
            resnet18_model = resnet18(ResNet18_Weights.DEFAULT).to(DEVICE)
            resnet18_model = nn.Sequential(*list(resnet18_model.children())[:-1]).to(
                DEVICE
            )
            resnet18_model.eval()
            return resnet18_model(image_batch).reshape(512)
        else:
            # change ckpt here
            vae_ckpt = "models/categorisation/bolt_VAE/version_0/checkpoints/epoch=46-step=128921.ckpt"
            model = VAE.load_from_checkpoint(vae_ckpt)
            model = nn.Sequential(*list(model.children())[0])
            model.eval()
            return model


def pca(samples: np.ndarray, n_components: int):
    decomp = PCA(n_components=n_components, copy=True)
    return decomp.fit_transform(samples)


def generate_hog_features(image: np.ndarray):
    if image.shape != (256, 256):
        raise ValueError(
            "Input image should have shape (256, 256), instead it is", image.shape
        )
    return hog(image, orientations=8, pixels_per_cell=(64, 64), cells_per_block=(2, 2))
