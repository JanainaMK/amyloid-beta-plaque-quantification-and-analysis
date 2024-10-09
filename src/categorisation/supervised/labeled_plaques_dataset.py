import os
from typing import Callable, List, Optional, Union

import h5py
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.categorisation.supervised.data_split import FoldIterable, get_val_train_indices
from src.categorisation.supervised.plaque_labels import (
    count_occurrences,
    equidistant_labels,
)


class LabeledPlaquesDatasetModule(Dataset):
    def __init__(
        self,
        labeled_images_path: str,
        split: str,
        transform: Callable,
        start: int,
        stop: int,
        file_names,
        local_indices,
        labels,
        num_copies,
        **kwargs,
    ):

        self.labeled_images_path = labeled_images_path
        self.split = split
        self.transform = transform
        self.num_copies = num_copies
        self.file_names = file_names
        self.local_indices = local_indices
        self.labels = labels
        self.file_names = self.file_names[start:stop]
        self.local_indices = self.local_indices[start:stop]
        self.labels = self.labels[start:stop]
        # count_occurrences(self.labels)
        self.labels = torch.tensor(self.labels).long()

        if self.split == "train":
            # augment data if train split
            self.num_samples = len(self.file_names) * self.num_copies
        else:
            self.num_samples = len(self.file_names)

        print("number of samples:", self.num_samples)

        print(f"done setting up supervised {split} dataset\n")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.split == "train":
            # idx of the correspoding image based on all samples
            img_idx = idx // self.num_copies
            # idx that specifies which augmentation operation to perform on the corresponding image
            augmentation_idx = idx % self.num_copies

            label, file_name, local_idx = (
                self.labels[img_idx],
                self.file_names[img_idx],
                self.local_indices[img_idx],
            )
            with h5py.File(
                os.path.join(self.labeled_images_path, file_name), "r"
            ) as file:
                img = file["plaques"][f"{local_idx}"]["plaque"][()]
                img = Image.fromarray(img.astype("uint8"))
                all_transforms = []
                if augmentation_idx > 0:
                    # apply augmentation operation
                    if augmentation_idx == 1:
                        tr = transforms.RandomHorizontalFlip(p=1.0)
                    elif augmentation_idx == 2:
                        tr = transforms.RandomVerticalFlip(p=1.0)
                    elif augmentation_idx == 3:
                        self.jitter_strength = 1.0
                        self.color_jitter = transforms.ColorJitter(
                            0.2 * self.jitter_strength,
                            0.2 * self.jitter_strength,
                            0.2 * self.jitter_strength,
                            0.0 * self.jitter_strength,
                        )
                        tr = transforms.RandomApply([self.color_jitter], p=1.0)
                    else:
                        tr = transforms.RandomApply(
                            [transforms.GaussianBlur(kernel_size=7)], p=1.0
                        )

                    all_transforms.append(tr)
                    if self.transform is not None:
                        all_transforms.append(self.transform)
                    final_transform = transforms.Compose(all_transforms)
                    img = final_transform(img)
                    # pass an invalid image area for augmented samples which are not the original image
                    area = -11
                else:
                    # maintain the same image
                    if self.transform is not None:
                        img = self.transform(img)

                    area = file["plaques"][f"{local_idx}"].attrs["area"]

                return img, label, [file_name, local_idx, area]
        else:
            img_idx = idx
            label, file_name, local_idx = (
                self.labels[img_idx],
                self.file_names[img_idx],
                self.local_indices[img_idx],
            )
            with h5py.File(
                os.path.join(self.labeled_images_path, file_name), "r"
            ) as file:
                img = file["plaques"][f"{local_idx}"]["plaque"][()]
                img = Image.fromarray(img.astype("uint8"))

                if self.transform is not None:
                    img = self.transform(img)

                area = file["plaques"][f"{local_idx}"].attrs["area"]

            return img, label, [file_name, local_idx, area]


class LabeledPlaquesDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_path: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        labeled_images_path: str,
        labeled_idx_file: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        start: int = 0,
        stop: int = 10**7,
        preprocess: Callable = None,
        val_size: int = 0,
        num_copies: int = 5,
        n_folds: int = 5,
        n_class_per_fold: int = 7,
        **kwargs,
    ):
        super().__init__()

        self.labeled_images_path = labeled_images_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.start = start
        self.stop = stop
        self.preprocess = preprocess
        self.val_size = val_size
        self.num_copies = num_copies

        with np.load(labeled_idx_file) as plaques_info:
            file_names = plaques_info["file_name"]
            local_indices = plaques_info["local_idx"]
            labels = plaques_info["label"]

        self.file_names = file_names
        self.local_indices = local_indices
        # ensure label values are equidistant to avoid problems during training
        self.labels = equidistant_labels(labels)

        self.fold_iterable = FoldIterable(
            labels, n_folds=n_folds, n_class_per_fold=n_class_per_fold
        )

        if not self.preprocess:
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((self.patch_size, self.patch_size)),
                    transforms.ToTensor(),
                ]
            )

    def setup(self, stage: Optional[str] = None) -> None:
        # get test and temporary train indices from next fold in iterable
        remaining_index, test_indices = next(self.fold_iterable)
        # get labels from indices
        remaining_labels = self.labels[remaining_index]
        # split the temporary train fold indices to another set of validation and train indices
        val_indices, train_indices = get_val_train_indices(
            remaining_labels, val_size=self.val_size
        )

        self.train_labels = self.labels[remaining_index][train_indices]
        self.train_file_names = self.file_names[remaining_index][train_indices]
        self.train_local_indices = self.local_indices[remaining_index][train_indices]

        self.val_labels = self.labels[remaining_index][val_indices]
        self.val_file_names = self.file_names[remaining_index][val_indices]
        self.val_local_indices = self.local_indices[remaining_index][val_indices]

        self.test_labels = self.labels[test_indices]
        self.test_file_names = self.file_names[test_indices]
        self.test_local_indices = self.local_indices[test_indices]

        self.all_labels = np.concatenate(
            (self.train_labels, self.val_labels, self.test_labels)
        )
        self.all_file_names = np.concatenate(
            (self.train_file_names, self.val_file_names, self.test_file_names)
        )
        self.all_local_indices = np.concatenate(
            (self.train_local_indices, self.val_local_indices, self.test_local_indices)
        )

        self.train_dataset = LabeledPlaquesDatasetModule(
            self.labeled_images_path,
            split="train",
            file_names=self.train_file_names,
            local_indices=self.train_local_indices,
            labels=self.train_labels,
            transform=self.preprocess,
            download=False,
            start=self.start,
            stop=self.stop,
            num_copies=self.num_copies,
        )

        self.val_dataset = LabeledPlaquesDatasetModule(
            self.labeled_images_path,
            split="val",
            file_names=self.val_file_names,
            local_indices=self.val_local_indices,
            labels=self.val_labels,
            transform=self.preprocess,
            download=False,
            start=self.start,
            stop=self.stop,
            num_copies=self.num_copies,
        )

        self.test_dataset = LabeledPlaquesDatasetModule(
            self.labeled_images_path,
            split="test",
            file_names=self.test_file_names,
            local_indices=self.test_local_indices,
            labels=self.test_labels,
            transform=self.preprocess,
            download=False,
            start=self.start,
            stop=self.stop,
            num_copies=self.num_copies,
        )

        self.all_dataset = LabeledPlaquesDatasetModule(
            self.labeled_images_path,
            split="all",
            file_names=self.all_file_names,
            local_indices=self.all_local_indices,
            labels=self.all_labels,
            transform=self.preprocess,
            download=False,
            start=self.start,
            stop=self.stop,
            num_copies=self.num_copies,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def all_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.all_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
