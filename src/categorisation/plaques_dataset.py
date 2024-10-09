import json
import os
from typing import Callable, List, Optional, Union

import h5py
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.categorisation.file_filter import valid_region


class PlaquesDatasetModule(Dataset):
    def __init__(
        self,
        all_images_path: str,
        file_names: list,
        split: str,
        transform: Callable,
        start: int,
        stop: int,
        **kwargs,
    ):

        self.all_images_path = all_images_path
        self.file_names = file_names[start:stop]
        self.transform = transform

        self.cumulative_n_plaques = []
        self.total_n_plaques = 0
        for _, filename in enumerate(self.file_names):
            with h5py.File(os.path.join(self.all_images_path, filename), "r") as file:
                curr_n_plaques = file.attrs["n_plaques"]
                self.total_n_plaques += curr_n_plaques
                self.cumulative_n_plaques.append(self.total_n_plaques)

        print("number of files:", len(self.file_names))
        print("number of plaques: ", self.total_n_plaques)
        print(f"done setting up {split} dataset\n")

    def __len__(self):
        return self.total_n_plaques

    def __getitem__(self, idx):
        for i, c_size in enumerate(self.cumulative_n_plaques):
            if idx < c_size:
                # Calculate new indices based on cumulative number of plaques
                if i > 0:
                    prev_c_size = self.cumulative_n_plaques[i - 1]
                else:
                    prev_c_size = 0
                local_img_idx = idx - prev_c_size
                file_name = self.file_names[i]

                with h5py.File(
                    os.path.join(self.all_images_path, file_name), "r"
                ) as file:
                    try:
                        img = file["plaques"][f"{local_img_idx}"]["plaque"][()]
                        area = file["plaques"][f"{local_img_idx}"].attrs["area"]
                    except Exception as e:
                        print(
                            f"Error encountered while reading plaque in file {file_name}, local index {local_img_idx}."
                        )
                        raise e

                    img = Image.fromarray(img.astype("uint8"))
                orig_size = img.size
                if self.transform is not None:
                    img = self.transform(img)

                return img, [self.file_names[i], local_img_idx, area, orig_size]
        raise IndexError("Index out of bounds")


class PlaquesDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
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
        all_images_path: str,
        data_split_path: str,
        model_type: str = "moco",
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        start: int = 0,
        stop: int = 10**7,
        preprocess: Callable = None,
        **kwargs,
    ):
        super().__init__()
        self.all_images_path = all_images_path
        self.model_type = model_type
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.start = start
        self.stop = stop

        with open(data_split_path, "r") as json_file:
            train_test_split = json.load(json_file)

        self.train_file_names = train_test_split.get("train_files", [])
        self.val_file_names = train_test_split.get("test_files", [])
        self.test_file_names = train_test_split.get("test_files", [])

        self.train_file_names = [f for f in self.train_file_names if valid_region(f)]
        self.val_file_names = [f for f in self.val_file_names if valid_region(f)]
        self.test_file_names = [f for f in self.test_file_names if valid_region(f)]
        self.all_file_names = (
            self.train_file_names + self.val_file_names + self.test_file_names
        )

        self.train_file_names = sorted(list(set(self.train_file_names)))
        self.val_file_names = sorted(list(set(self.val_file_names)))
        self.test_file_names = sorted(list(set(self.test_file_names)))
        self.all_file_names = sorted(list(set(self.all_file_names)))

        self.train_file_names = self.train_file_names[start:stop]
        self.val_file_names = self.val_file_names[start:stop]
        self.test_file_names = self.test_file_names[start:stop]
        self.all_file_names = self.all_file_names[start:stop]

        flip_transform = transforms.RandomChoice(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )

        if not preprocess:
            if self.model_type == "simclr" or self.model_type == "moco":
                preprocess = transforms.Compose(
                    [
                        transforms.Resize((self.patch_size, self.patch_size)),
                        transforms.ToTensor(),
                    ]
                )
            else:
                preprocess = transforms.Compose(
                    [
                        flip_transform,
                        transforms.Resize((self.patch_size, self.patch_size)),
                        transforms.ToTensor(),
                    ]
                )
        self.preprocess = preprocess

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = PlaquesDatasetModule(
            self.all_images_path,
            file_names=self.train_file_names,
            split="train",
            transform=self.preprocess,
            download=False,
            start=self.start,
            stop=self.stop,
        )

        self.val_dataset = PlaquesDatasetModule(
            self.all_images_path,
            file_names=self.val_file_names,
            split="val",
            transform=self.preprocess,
            download=False,
            start=self.start,
            stop=self.stop,
        )

        self.test_dataset = PlaquesDatasetModule(
            self.all_images_path,
            file_names=self.test_file_names,
            split="test",
            transform=self.preprocess,
            download=False,
            start=self.start,
            stop=self.stop,
        )

        self.all_dataset = PlaquesDatasetModule(
            self.all_images_path,
            file_names=self.all_file_names,
            split="all",
            transform=self.preprocess,
            download=False,
            start=self.start,
            stop=self.stop,
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
            batch_size=self.train_batch_size,
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
