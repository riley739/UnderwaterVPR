from typing import Optional, Callable, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

from PIL import Image



default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



class BaseValDataset(Dataset):
    def __init__(
        self,
        name,
        path,
        config
    ):
        self.input_transform = default_transform

        self.dataset_name = name
                
        self.dataset_path = Path(path)

        # Load image names and ground truth data
        self.dbImages = np.load(self.dataset_path / f"db_images.npy")
        self.qImages = np.load(self.dataset_path / f"q_images.npy")
        self.ground_truth = np.load(self.dataset_path / f"gt.npy", allow_pickle=True)

        # Combine reference and query images
        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index: int, return_path = False) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, index) where image is a PIL image.
        """
        img_path = self.image_paths[index]
        img = Image.open(self.dataset_path / img_path)

        if return_path:
            return img, return_path

        if self.input_transform:
            img = self.input_transform(img)

       
        return img, index, 

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset.
        """
        return len(self.image_paths)

    def set_transform(self, transform):
        self.input_transform = transform
