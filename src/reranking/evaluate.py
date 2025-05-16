# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch

from src.core.datamodule import DataModule
from src.utils.modules_manager import create_model, load_checkpoint
from src.utils.config_manager import parse_args
#TODO: Move into own folder
from visualize_temp import save_preds
# from src.utils.visualize_cameras import save_cameras
import sys
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
from torchvision.transforms import v2  as T2
import threading
from lcm_src.lcm_handler import LCMImageReceiver

import os
import sys

# Get the absolute path to the `main/` directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

val_transform = T2.Compose([
            T2.ToImage(),
            T2.Resize(size=322, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def get_descriptors(model, all_descriptors, dataloader, device):
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            images, indices = batch
            descriptors = model(images.to(device))
            if isinstance(descriptors, tuple) or isinstance(descriptors, list):
                output = descriptors[0].detach().cpu()
            else:
                output = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = output
    
    return all_descriptors

def get_descriptor(model, image, device):
    with torch.inference_mode():
        descriptor = model(image.to(device))

        if isinstance(descriptor, tuple) or isinstance(descriptor, list):
            output = descriptor[0].detach().cpu()
        else:
            output = descriptor.cpu().numpy()

        return output


def setup_model(config):
    evaluation_params = config["evaluation"]
    model = create_model(config)

    load_checkpoint(model, evaluation_params["checkpoint_path"])

    logger.info(f"Checkpoint loaded from {evaluation_params["checkpoint_path"]}")

    model = model.eval().to(config["device"])

    datamodule  = DataModule(config["datamodule"])

    datamodule.setup("test")

    return model, datamodule

def load_descriptors(model, dataset, config):


    logger.info(f"Testing on {dataset}")

    database_dataloader = DataLoader(
        dataset=dataset, num_workers=config["datamodule"]["num_workers"], batch_size=config["datamodule"]["batch_size"]
    )

    #TODO: Move from val to test datasets here... 
    #TODO: Split into database and query datasets... 

    logger.debug("Extracting descriptors for evaluation/testing")
    #TODO: Check if passing the all_descriptors as key value ruins the speed up of creating the nump array 
    all_descriptors = np.empty((len(dataset), config["evaluation"]["descriptor_dimension"]), dtype="float32")
    all_descriptors = get_descriptors(model, all_descriptors, database_dataloader, config["device"])

    query_descriptors = all_descriptors[dataset.num_references :]
    database_descriptors = all_descriptors[: dataset.num_references]

    return query_descriptors, database_descriptors

    

def get_correct(predictions, dataset, pose, theshold = 1):

    correct = []
    
    for pred in predictions[0]:
        try:
            correct.append(dataset.is_correct(pred, pose, theshold))
        # IF not associated pose just assume its wrong
        except:
            correct.append(False)   
    return correct

# This is called when the train mode is selected
def evaluate(config):
    start_time =  datetime.now()
    log_dir = Path("logs") / config["evaluation"]["log_dir"] / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")


    model, datamodule = setup_model(config)

    #TODO Work with multiple datasets e.g. append them 
    dataset = datamodule.get_datasets("test")[0]
    query_descriptors, database_descriptors = load_descriptors(model, dataset, config )

    
    torch.save(query_descriptors, "queries.pt")
    torch.save(database_descriptors, "database.pt")


    


    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(config["evaluation"]["descriptor_dimension"])
    faiss_index.add(database_descriptors)

    logger.debug("Calculating recalls")
    _, predictions = faiss_index.search(query_descriptors, max(config["evaluation"]["recall_values"]))
    correct_at_k = np.zeros(len(config["evaluation"]["recall_values"]))

    _, predictions = faiss_index.search(query_descriptors, max(config["evaluation"]["recall_values"]))

    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(config["evaluation"]["recall_values"]):
                # if in top N then also in top NN, where NN > N
                if dataset.is_correct(q_idx, pred[:n]):
                    correct_at_k[i:] += 1
                    break
    correct_at_k = correct_at_k / len(predictions)
    d = {k:v for (k,v) in zip(config["evaluation"]["recall_values"], correct_at_k)}

    table = PrettyTable()
    table.field_names = ['K']+[str(k) for k in config["evaluation"]["recall_values"]]
    table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
    logger.info(table.get_string(title=f"Performance on {dataset.dataset_name}"))


    # topk = q_idx in query_descp -> database_vectors[pred[:k]]

def rerank():
    config = parse_args()
    
    evaluate(config)
