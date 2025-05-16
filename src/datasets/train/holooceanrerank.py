from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np 
import faiss 
import pandas as pd 
from src.reranking.fov import fov2d_overlap_pairs

# Now we can define the dataset class
#TODO: Process seems to be working quite ok  tho majorly overtrained not thats that too unexpected
# add affinities next 
class HoloOceanRerankTrainDataset(Dataset):
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
        self.dataset_name = name

        #get dataframes (this is what we will need in the end not the others) 
        self.db = pd.read_csv(self.base_path / f"db.csv")
        self.q = pd.read_csv(self.base_path / f"query.csv")

        self.dbImages, self.qImages, self.ground_truth = self.get_images()

        self.query_desc = torch.load(self.base_path / f"queries.pt", weights_only=False)
        self.db_desc = torch.load(self.base_path / f"database.pt", weights_only=False)

        # Use a kNN to find predictions
        #TODO Update the descriptor dimension location - assert both descs are the same and then set jere
        faiss_index = faiss.IndexFlatL2(12288)
        faiss_index.add(self.db_desc)

        #TODO: Update the number of things searched for 
        _, predictions = faiss_index.search(self.query_desc, 100)
        
        #TODO Come from something else.
        self.affinity = ["positional"]
        self.db_coords = self.db[['x', 'y']].to_numpy()
        self.query_coords = self.q[['x', 'y']].to_numpy()
        
        self.predictions = predictions
        self.topk = self.db_desc[predictions]

        #TODO: THese are only needed in the val dataset
        self.total_nb_images = len(self)
        self.num_queries = self.query_desc.shape[0]
        self.num_references = self.db_desc.shape[0]

    def _affinity(self, query_idx: int):
        aff = []
        predictions = self.predictions[query_idx]

        for a in self.affinity:
            if a == "positional":
                aff.append(torch.from_numpy(fov2d_overlap_pairs(self.db_coords[predictions], np.zeros((len(self.db_coords[predictions]), 1)))))
        return aff
    
    def _labels(self, query_idx):
        predictions = self.predictions[query_idx]
        label = self.ground_truth[query_idx]

        labels_np = np.in1d(predictions, label) 

        labels = torch.from_numpy(labels_np)

        labels = torch.cat((torch.ones(1,dtype=bool), labels))

        return labels

    def __getitem__(self, index):
        query_desc = torch.from_numpy(self.query_desc[index]).unsqueeze(0)
        db_desc = torch.from_numpy(self.topk[index])

        desc = torch.cat((query_desc, db_desc), dim=0)

        labels = self._labels(index)
        aff = self._affinity(index)


        db_idx = torch.cat((torch.tensor([index]), torch.from_numpy(self.predictions[index])), dim=0)


        return desc, labels, [],  db_idx

    def __len__(self) -> int:
        """! Get number of items.

        @return The number of items in the dataset.
        """
        return self.query_desc.shape[0]

    def set_transform(self, transform):
        return


    def is_correct(self, index: int, predictions) -> bool:
        label = self.ground_truth[index]
        if np.any(np.in1d(predictions, label)):
            return True
        return False
    

    def get_images(self):

        db_images = self.db["name"].tolist()
        db_images = [self.base_path / "Images" / db for db in db_images]
        query_images = self.q["name"].tolist() 
        query_images = [self.base_path / "Images" / query for query in query_images]


        ground_truths = [] 

        for _, row in self.q.iterrows():
            place_id = row["place_id"]
            matching_idx = self.db.index[self.db["place_id"] == place_id].tolist()
            ground_truths.append(matching_idx)

   
        return db_images, query_images, ground_truths 
