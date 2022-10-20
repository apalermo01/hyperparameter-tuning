from typing import Union
import pytorch_lightning as pl
from torchvision import datasets as ds
from torch.utils.data import Dataset
from torchvision import transforms
from hparam_tuning_project.utils import PATHS
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
import os


class PytorchDataset(pl.LightningDataModule):
    """Dataset wrapper for the default pytorch datasets

    TODO:
    restrict dataset size
    """

    dataset_registry = {
        'caltech_101': ds.Caltech101,
        'caltech_256': ds.Caltech256,
        'celeba': ds.CelebA,
        'cifar10': ds.CIFAR10,
        'cifar100': ds.CIFAR100,
        'country211': ds.Country211,
        'emnist': ds.EMNIST,
        'eurosat': ds.EuroSAT,
        'fake_data': ds.FakeData,
        'fashion_mnist': ds.FashionMNIST,
        'fer2013': ds.FER2013,
        'fgvc_aircraft': ds.FGVCAircraft,
        'imagenet': ds.ImageNet,
        'mnist': ds.MNIST,
    }

    def __init__(self,
                 dataset_id: str,
                 train: bool,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 use_default_path: bool = True,
                 train_split_size: float = 0.8,
                 dataset_path: Union[str, None] = None,
                 use_precomputed_split: bool = True,
                 split_id: Union[str, None] = None,
                 split_path: Union[str, None] = None):
        super().__init__()

        self.dataset_id = dataset_id
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train = train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split_size = train_split_size
        self.use_precomputed_split = use_precomputed_split
        self.split_id = split_id
        self.split_path = split_path
        if use_default_path:
            self.dataset_path = PATHS['dataset_path'] + dataset_id + "/"
        else:
            assert dataset_path is not None, "if you don't want to use the default path, you should pass the path you want to dataset_path"
            self.dataset_path = dataset_path

        self.vision_dataset = self.dataset_registry[self.dataset_id](
            root=self.dataset_path,
            train=self.train,
            download=True,
            transform=self.transform,
            target_transform=None,
        )

    def setup(self, stage=None):

        train_val = self.dataset_registry[self.dataset_id](
            root=self.dataset_path,
            train=True,
            transform=self.transform
        )

        if self.use_precomputed_split:
            if self.split_id is None:
                split_id = self.dataset_id
            else:
                split_id = self.split_id

            if self.split_path is None:
                split_path = "./splits/"
            else:
                split_path = self.split_path

            train_idx = np.loadtxt(os.path.join(split_path, f"{split_id}_train.txt"))
            val_idx = np.loadtxt(os.path.join(split_path, f"{split_id}_val.txt"))
            self.train_dataset = Subset(train_val, train_idx.astype(int))
            self.val_dataset = Subset(train_val, val_idx.astype(int))
            # print(self.train_dataset)
        else:
            train_size = int(self.train_split_size * len(train_val))
            val_size = len(train_val) - train_size
            self.train_dataset, self.val_dataset = random_split(train_val, [train_size, val_size])
        # print(self.train_dataset)
        # assert False

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            sampler=None,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

        return train_loader

    def val_dataloader(self):
        train_loader = DataLoader(
            dataset=self.val_dataset,
            sampler=None,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

        return train_loader
