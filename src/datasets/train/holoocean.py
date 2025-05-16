# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
from src.datasets.train.base_dataset import BaseTrainDataset

# Now we can define the dataset class
class HoloOceanTrainDataset(BaseTrainDataset):
    def __init__(self,
                 **kwargs
                 ):
        """
        Args:
            cities (list): List of city names to use in the dataset. Default is "all" or None which uses all cities.
            base_path (Path): Base path for the dataset files.
            img_per_place (int): The number of images per place.
            random_sample_from_each_place (bool): Whether to sample images randomly from each place.
            transform (callable): Optional transform to apply on images.
            hard_mining (bool): Whether you are performing hard negative mining or not.
        """
        super().__init__(**kwargs)

    def getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images

            This requiers DataFrame files to be in a folder
            named Dataframes
        '''
        df = pd.read_csv(self.base_path /'Dataframes/Database.csv')
       
        self.coords =  df[['x', 'y', 'z']].to_numpy()
        # keep only places depicted by at least img_per_place images
        return df
        
    def __getitem__(self, index):
        anchor_row = self.dataframe.iloc[index]
        anchor_coord = self.coords[index]
        

        # Compute distances to all other points
        dists = np.linalg.norm(self.coords - anchor_coord, axis=1)
        #TODO: Update this to be distance threshold
        nearby_idxs = np.where(dists < 10)[0]

        # Fallback if not enough nearby
        if len(nearby_idxs) < self.img_per_place:
            # fallback to closest K
            nearby_idxs = np.argsort(dists)[:self.img_per_place]
        else:
            nearby_idxs = np.random.choice(nearby_idxs, self.img_per_place, replace=False)

        imgs = []
        for idx in nearby_idxs:
            row = self.dataframe.iloc[idx]
            img_path = self.base_path / 'Images' /  "Database" / row['name']  # make sure 'image_path' exists
            img = self.image_loader(img_path)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # Assign unique label for this group (can just use index to ensure uniqueness)
        label = torch.tensor(index).repeat(self.img_per_place)

        return torch.stack(imgs), label
    

