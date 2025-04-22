from typing import Optional, Callable, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from src.utils.transforms import default_transform

from PIL import Image

class BaseTestDataset(Dataset):
    def __init__(
        self,
        name,
        path,
        config
    ):
        
        #TODO This should be changed to come from config if given
        self.input_transform = default_transform

        self.dataset_name = name
                
        self.dataset_path = Path(path)

        self.dbImages, self.qImages, self.ground_truth = self.get_images()
        # Load image names and ground truth data
 

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
        img = Image.open(self.dataset_path / img_path).convert("RGB")
        
        if self.input_transform:
            img = self.input_transform(img)

        if return_path:
            return img, return_path
        
        return img, index, 

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset.
        """
        return len(self.image_paths)

    def set_transform(self, transform):
        self.input_transform = transform

    def is_correct(self, index: int, predictions) -> bool:
        label = self.ground_truth[index]
        if np.any(np.in1d(predictions, label)):
            return True
        return False
    
    
    def get_images(self):
        dbImages = np.load(self.dataset_path / f"db_images.npy")
        qImages = np.load(self.dataset_path / f"q_images.npy")
        ground_truth = np.load(self.dataset_path / f"gt.npy", allow_pickle=True)

        return dbImages, qImages, ground_truth
