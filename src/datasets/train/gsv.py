# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

"""
GSV-Cities dataset 
====================

This module implements a PyTorch Dataset class for GSV-Cities dataset from the paper:

"GSV-Cities: Toward Appropriate Supervised Visual Place Recognition" 
by Ali-bey et al., published in Neurocomputing, 2022.


Citation:
    @article{ali2022gsv,
        title={{GSV-Cities}: Toward appropriate supervised visual place recognition},
        author={Ali-bey, Amar and Chaib-draa, Brahim and Gigu{\`e}re, Philippe},
        journal={Neurocomputing},
        volume={513},
        pages={194--203},
        year={2022},
        publisher={Elsevier}
    }

URL: https://arxiv.org/abs/2210.10239
"""

import pandas as pd
from src.datasets.train.base_dataset import BaseTrainDataset

# Now we can define the dataset class
class GSVCitiesTrainDataset(BaseTrainDataset):
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
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing one DataFrame
            for each city in self.cities
        '''
        self.cities = [f.name[:-4] for f in self.base_path.glob("Dataframes/*.csv")]

        dataframes = []
        for i, city in enumerate(self.cities):
            df = pd.read_csv(self.base_path / 'Dataframes' / f'{city}.csv')
            df['place_id'] += i * 10**5 # to avoid place_id conflicts between cities
            df = df.sample(frac=1) # we always shuffle in city level
            dataframes.append(df)
        
        df = pd.concat(dataframes)
        # keep only places depicted by at least img_per_place images
        df = df[df.groupby('place_id')['place_id'].transform('size') >= self.img_per_place]
        return df.set_index('place_id')
        
    @staticmethod
    def get_img_name(row):
        """
            Given a row from the dataframe
            return the corresponding image name
        """
        city = row['city_id']
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
        return name

