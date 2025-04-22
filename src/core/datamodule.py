# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import lightning as L
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torchvision.transforms import v2  as T2

from src.utils.modules_manager import get_dataset



#TODO:Figure out if need hard mining
class DataModule(L.LightningDataModule):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.batch_size = config["batch_size"]
        self.shuffle_all = config.get("shuffle_all", False)
        self.train_image_size = config["train_image_size"]
        self.val_image_size = config.get("val_image_size", self.train_image_size)
        self.num_workers = config.get("num_workers", 4)
        self.mean_std = config.get("mean_std", {"mean":[0.485, 0.456, 0.406], "std":[0.229, 0.224, 0.225]})
        self.val_set_names = config["val_set_names"]
        self.test_set_names = config.get("test_set_names", [])
        self.train_set_name = config["train_set_name"]
        self.img_per_place = config["img_per_place"] # Needed for callback
    
        self.config = config

        # Define the train transformations
        self.train_transform = T2.Compose([
            T2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            T2.Resize(size=self.train_image_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.RandAugment(num_ops=3, magnitude=15, interpolation=T2.InterpolationMode.BILINEAR),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
        ])

        # Define the validation transformations
        self.val_transform = T2.Compose([
            T2.ToImage(),
            T2.Resize(size=self.val_image_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
        ])

        # Define the test transformations
        self.test_transform = T2.Compose([
            T2.ToImage(),
            T2.Resize(size=self.val_image_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=self.mean_std["mean"], std=self.mean_std["std"]),
        ])

        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None
        
    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = self._get_train_dataset()
            self.val_datasets = [self._get_val_dataset(ds_name) for ds_name in self.val_set_names]
        if stage == "predict":
            self.val_datasets = [self._get_val_dataset(ds_name) for ds_name in self.val_set_names]
        if stage == "test":
            self.test_datasets = [self._get_test_dataset(ds_name) for ds_name in self.test_set_names]
               
    def train_dataloader(self):
        # the reason we are using `_get_train_dataset` here is because
        # sometimes we want to shuffle the data (in-city only) at each epoch
        # which can only be done when loading the dataset's dataframes
        self.train_dataset = self._get_train_dataset()
        
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=self.shuffle_all,
        )

    def val_dataloader(self):
        val_dataloaders = []
        for i,dataset in enumerate(self.val_datasets):
            dl = DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=False,
                    pin_memory=True,
                    shuffle=False
                )
            val_dataloaders.append(dl)
        return val_dataloaders
    
    def get_datasets(self, type):
        if type == "train":
            return self.train_dataset
        elif type == "val":
            return self.val_datasets
        elif type == "test":
            return self.test_datasets
        else:
            raise ValueError(f"Unknown type {type}")
    
    def _get_train_dataset(self):
        
        training_dataset = get_dataset(self.train_set_name, self.config, "train")
        training_dataset.set_transform(self.train_transform)

        return training_dataset
        
    def _get_val_dataset(self, ds_name):  

        val_dataset = get_dataset(ds_name, self.config, "val")
        val_dataset.set_transform(self.val_transform)

        return val_dataset
    
    def _get_test_dataset(self, ds_name):  

        val_dataset = get_dataset(ds_name, self.config, "test")
        val_dataset.set_transform(self.test_transform)

        return val_dataset

