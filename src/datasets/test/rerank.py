import pandas as pd 
from src.datasets.test.base_dataset import BaseTestDataset
from PIL import Image

class RerankTestDataset(BaseTestDataset):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.db = pd.read_csv(self.dataset_path / f"db.csv")
        self.query = pd.read_csv(self.dataset_path / f"query.csv")

        self.db_coords = self.db[['x', 'y', 'z']].to_numpy()
        self.query_coords = self.query[['x', 'y', 'z']].to_numpy()

    def __getitem__(self, index: int, return_path = False):
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
    
    def get_images(self):
        db = pd.read_csv(self.dataset_path / f"db.csv")
        query = pd.read_csv(self.dataset_path / f"query.csv")

        db_images = db["name"].tolist()
        db_images = [self.dataset_path / "Images" / db for db in db_images]
        query_images = query["name"].tolist() 
        query_images = [self.dataset_path / "Images" / query for query in query_images]


        ground_truths = [] 

        for _, row in query.iterrows():
            place_id = row["place_id"]
            matching_idx = db.index[db["place_id"] == place_id].tolist()
            ground_truths.append(matching_idx)

   
        return db_images, query_images, ground_truths 


