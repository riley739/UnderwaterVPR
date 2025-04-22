import glob
from src.datasets.test.base_dataset import BaseTestDataset
import numpy as np

class HoloOceanLiveTestDataset(BaseTestDataset):
    def __init__(
        self,
        name,
        path,
        config
    ):
        super().__init__(name, path, config)



    #TODO Based on pose of image 
    def is_correct(self, index: int, predictions) -> bool:
        return False
    
    
    def get_images(self):
        image_paths = []
        for path in glob.glob(f"{self.dataset_path}/Images/*"):
            image_paths.append(path)

        # No reference or ground truths
        return image_paths, [], [] 
