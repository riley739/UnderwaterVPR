from typing import Optional, Callable, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
import pandas as pd

from PIL import Image



default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#TODO make this inherit from base_dataset 
class HoloOceanDistanceValDataset(Dataset):
    def __init__(
        self,
        name,
        path,
        config
    ):
        self.input_transform = default_transform

        self.dataset_name = name
                
        self.dataset_path = Path(path)
        #TODO COnvert this to dataframes and yeah  
        # Load image names and ground truth data
        self.dbImages = pd.read_csv(self.dataset_path / f"db_images.csv")
        self.qImages = pd.read_csv(self.dataset_path / f"q_images.csv")

        self.coords = self.dbImages[['x', 'y', 'z']].to_numpy()

        # Combine reference and query images
        image_paths = np.concatenate((self.dbImages['name'].to_numpy(), self.qImages['name'].to_numpy())).astype(str)
        self.image_paths = np.char.add('Images/', image_paths)
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


    def is_correct(self, index: int, predictions) -> bool:
        item = self.qImages.iloc[index]

        position = [item['x'], item['y'], item['z']]
        coords = self.coords[predictions]
        dists = np.linalg.norm(coords - position, axis=1)

        #TODO: Update this to be distance threshold, probs come through the function i think? gets calld in two places thos so watchout
        nearby_idxs = np.where(dists < 10)[0]

        if len(nearby_idxs) > 0:
            return True
        return False
