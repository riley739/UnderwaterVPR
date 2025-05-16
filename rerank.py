# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch

from src.core.datamodule import DataModule
from src.utils.config_manager import parse_args
from src.utils.modules_manager import create_model
# from src.utils.visualize_cameras import save_cameras
import sys
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

import faiss
import numpy as np
from loguru import logger
from prettytable import PrettyTable
from tqdm import tqdm

import os
import sys

# Get the absolute path to the `main/` directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)



def setup_dataset(config):

    model = create_model(config)

    model = model.eval().to(config["device"])

    datamodule  = DataModule(config["datamodule"])

    #TODO: Change this to test
    datamodule.setup("test")

    return model, datamodule

def get_descriptors(model, all_descs, dataloader, device ="cuda"):
    all_descs = []
    all_labels = []
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            descs, labels, affs = batch
            descs = model(descs.to(device)) 



            all_descs.append(descs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        descs = np.concatenate(all_descs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return descs, labels

def load_descriptors(model, dataset, config ):
    database_dataloader = DataLoader(
        dataset=dataset, num_workers=config["datamodule"]["num_workers"], batch_size=config["datamodule"]["batch_size"]
    )

    all_descs = np.empty((len(dataset), config["evaluation"]["descriptor_dimension"]), dtype="float32")
    descs, labels = get_descriptors(model, all_descs, database_dataloader, config["device"])

    return descs, labels

# This is called when the train mode is selected
def evaluate(config):

    model, datamodule = setup_dataset(config)

    dataset = datamodule.get_datasets("test")[0]

    descs, predictions = load_descriptors(model, dataset, config)

    correct_at_k = np.zeros(len([1,5,10,25]))
    #TODO: THis neesd to be updated to use the dataset class
    for q_idx,pred in enumerate(predictions): #Skip first element as its query
        for i, n in enumerate([1,5,10,25]):
                # if in top N then also in top NN, where NN > N
                if np.any(pred[1:n+1] == 1):
                    correct_at_k[i:] += 1
                    break
    correct_at_k = correct_at_k / len(predictions)

    d = {k:v for (k,v) in zip([1,5,10,25], correct_at_k)}

    table = PrettyTable()
    table.field_names = ['K']+[str(k) for k in [1,5,10,25]]
    table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
    logger.info(table.get_string(title=f"Performance on {dataset.dataset_name}"))

    # # Use a kNN to find predictions
    # faiss_index = faiss.IndexFlatL2(config["evaluation"]["descriptor_dimension"])
    # faiss_index.add(database_descriptors)

    # _, predictions = faiss_index.search(query_descriptors, 100)

    # #TODO: change k to come from config
    # topk = torch.empty((query_descriptors.shape[0], 100), dtype=torch.int32)

    # topk = database_descriptors[predictions]

    #The above dataset creation happens at run time.. then the model trains on this, makes sense.. 
    # might be able to use framework but doesn't have a backbone / aggregator -> could do if else ?? 

    # __get_item__ returns  -> [query_desc, db_desc] , labels [can try labels or 0/1s] , affinities 
    # Then put into model and train? 
    # for ind,query in enumerate(query_descriptors):
    #     query_desc = query 
    #     db_desc = topk[ind]


    #Can use framework -> pass aggregator as normal -> backbone is just a model that returns x.





    # topk = q_idx in query_descp -> database_vectors[pred[:k]]



if __name__ == "__main__":
    config = parse_args()
    from src.reranking.train import train
    from src.reranking.evaluate import evaluate

    train(config)
    
    # evaluate(config)
