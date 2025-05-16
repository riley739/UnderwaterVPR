import glob
from src.datasets.test.base_dataset import BaseTestDataset
import numpy as np
import pandas as pd 

#NEed to go off dataframes, gets the pose, checks within threshold return true or false 
class HoloOceanLiveTestDataset(BaseTestDataset):
    def __init__(
        self,
        name,
        path,
        config
    ):
        super().__init__(name, path, config)



    #TODO Maybe set the theshold elsewehere
    
    def is_correct(self, index: int, predictions, threshold = 1) -> bool:
        position = self.coords[index]
        dx = position[0] - predictions[0]
        dy = position[1] - predictions[1]
        return dx * dx + dy * dy <= threshold * threshold


    def get_images(self):
        df = pd.read_csv(self.dataset_path /'holooceanLive.csv')
        self.coords =  df[['x', 'y', 'z']].to_numpy()

        dbImages = df["name"].to_numpy().astype("str")
        dbImages = np.char.add('Images/', dbImages)

        return dbImages , [], []  