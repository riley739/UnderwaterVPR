from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np 
import faiss 
import pandas as pd 

# Now we can define the dataset class
#TODO: Process seems to be working quite ok  tho majorly overtrained not thats that too unexpected
# add affinities next 
class RerankingTrainDataset(Dataset):
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
        # self.db_dataframe = pd.read_csv(self.base_path / f"db.csv")
        # self.query_dataframe = pd.read_csv(self.base_path / f"query.csv")

    
        # # generate the dataframe contraining images metadata
        self.dbImages = np.load(self.base_path / f"db_images.npy")
        self.qImages = np.load(self.base_path / f"q_images.npy")
        self.ground_truth = np.load(self.base_path / f"gt.npy", allow_pickle=True)

        self.query_desc = torch.load(self.base_path / f"queries.pt", weights_only=False)
        self.db_desc = torch.load(self.base_path / f"database.pt", weights_only=False)

        # Use a kNN to find predictions
        #TODO Update the descriptor dimension location - assert both descs are the same and then set jere
        faiss_index = faiss.IndexFlatL2(12288)
        faiss_index.add(self.db_desc)

        #TODO: Update the number of things searched for 
        _, predictions = faiss_index.search(self.query_desc, 100)

        # self.query_heading = get_headings(self.query_dataframe, self.qImages)
        # self.db_heading = get_headings(self.db_dataframe, self.dbImages)

        
        self.predictions = predictions
        self.topk = self.db_desc[predictions]

        #TODO: THese are only needed in the val dataset
        self.total_nb_images = len(self)
        self.num_queries = self.query_desc.shape[0]
        self.num_references = self.db_desc.shape[0]

    # def get_positions
    # def _affinity(self, query_idx: int, db_idx: np.ndarray):
    #     aff = []

    #     for a in self.affinity:
    #         if a == "positional-db":
    #             aff.append(torch.from_numpy(fov2d_overlap_pairs(self.db_coords[db_idx], self.db_heading[db_idx])))
    #         elif a in ("heading", "heading-db"):
    #             heading = self.db_heading[db_idx]
    #             if a == "heading":
    #                 heading = np.concatenate([[self.query_heading[query_idx]], heading])
    #             heading_diff = np.abs(heading - heading[None].T)
    #             heading_diff = np.minimum(heading_diff, 2 * np.pi - heading_diff)
    #             aff.append(torch.from_numpy((np.pi - 2 * heading_diff) / np.pi).float())
    #         else:
    #             raise ValueError(f"Invalid affinity type {a}")

    #     return aff
    
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
        aff = [] 

        db_idx = torch.cat((torch.tensor([index]), torch.from_numpy(self.predictions[index])), dim=0)


        return desc, labels, aff,  db_idx

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
    