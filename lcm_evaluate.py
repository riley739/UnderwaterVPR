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
import cv2

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

def load_database_descriptors(model, dataset, config):


    logger.info(f"Testing on {dataset}")

    database_dataloader = DataLoader(
        dataset=dataset, num_workers=config["datamodule"]["num_workers"], batch_size=config["datamodule"]["batch_size"]
    )

    #TODO: Move from val to test datasets here... 
    #TODO: Split into database and query datasets... 

    logger.debug("Extracting database descriptors for evaluation/testing")
    #TODO: Check if passing the all_descriptors as key value ruins the speed up of creating the nump array 
    all_descriptors = np.empty((len(dataset), config["evaluation"]["descriptor_dimension"]), dtype="float32")
    all_descriptors = get_descriptors(model, all_descriptors, database_dataloader, config["device"])

    database_descriptors = all_descriptors[: dataset.num_references]

    return database_descriptors


def get_correct(predictions, dataset, pose, theshold = 1):

    correct = []
    
    for pred in predictions[0]:
        try:
            correct.append(dataset.is_correct(pred, pose, theshold, True))
        # IF not associated pose just assume its wrong
        except:
            correct.append((False, 0))   
    return correct

# This is called when the train mode is selected
def evaluate(config):
    start_time =  datetime.now()
    log_dir = Path("logs") / config["evaluation"]["log_dir"] / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")


    model, datamodule = setup_model(config)

    #TODO WOrk with multiple datasets e.g. append them 
    dataset = datamodule.get_datasets("test")[0]
    database_descriptors = load_database_descriptors(model, dataset, config )
    
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(config["evaluation"]["descriptor_dimension"])
    faiss_index.add(database_descriptors)
    del database_descriptors


    receiver = LCMImageReceiver()
    receiver.start_background_thread()

    count = 0 
    while True:
        img = receiver.get_latest_image()
        if img is not None:
            pose = receiver.get_latest_pose() #TODO : This should line up relatively well but should be checked as pose and camera are sent differently , pose is more common tho so should give relatively accurate estimation
            img_tensor = val_transform(img).unsqueeze(0)
            query_descriptor = get_descriptor(model, img_tensor, config["device"])

            logger.debug("Calculating recalls")
            _, predictions = faiss_index.search(query_descriptor, max(config["evaluation"]["recall_values"]))

            predictions = predictions[:, : config["evaluation"]["num_preds_to_save"]]

            threshold = 5
            correct = get_correct(predictions, dataset, pose, threshold)
            img = save_preds(predictions, correct, dataset, log_dir, img, count)
            count += 1



if __name__ == "__main__":
    config = parse_args()
    
    evaluate(config)
