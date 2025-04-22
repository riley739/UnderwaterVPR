import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.utils.transforms import default_transform

# Now we can define the dataset class
class BaseTrainDataset(Dataset):
    def __init__(self,
                 name,
                 path,
                 config
                 ):
        """
        Args:
            path: Path to the dataset folder.
            config: config dictionary containing the settings.
        """
        super().__init__()
        
        self.base_path = Path(path)
        self.name = name
        
        self.img_per_place = config["img_per_place"]
        self.random_sample_from_each_place = config.get("random_sample_from_each_place", True)
        self.transform = default_transform

        # generate the dataframe contraining images metadata
        self.dataframe = self.getdataframes()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
    def getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images

            This requiers DataFrame files to be in a folder
            named Dataframes
        '''
        df = pd.read_csv(self.base_path / 'Dataframes' / f'{self.name}.csv')
        # keep only places depicted by at least img_per_place images
        df = df[df.groupby('place_id')['place_id'].transform('size') >= self.img_per_place]
        return df.set_index('place_id')
        
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

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)
    
    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        """
            Given a row from the dataframe
            return the corresponding image name
        """
        return  row['image_name']

    def set_transform(self, transform):
        self.transform = transform
