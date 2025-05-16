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


            
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample k images
        place = place.sample(n=self.img_per_place) 
            
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path / 'Images' /  img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

            #TODO: Also need to return values for affinity here..

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)
