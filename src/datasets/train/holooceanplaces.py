import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from src.datasets.train.base_dataset import BaseTrainDataset
import torchvision.transforms as T


# Now we can define the dataset class
class HoloOceanPlacesTrainDataset(BaseTrainDataset):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs)


    