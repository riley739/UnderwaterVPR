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
from src.utils.visualize_places import save_preds
from src.utils.visualize_cameras import save_cameras
import sys
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable


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

# This is called when the train mode is selected
def evaluate(config):

    start_time = datetime.now()
    evaluation_params = config["evaluation"]

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / evaluation_params["log_dir"] / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {config}")
    logger.info(
        f"Testing with {config['aggregator']['method']} with a {config["backbone"]["model"]} backbone and descriptors dimension {evaluation_params['descriptor_dimension']}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")


    model = create_model(config)

    load_checkpoint(model, evaluation_params["checkpoint_path"])

    logger.info(f"Checkpoint loaded from {evaluation_params["checkpoint_path"]}")

    model = model.eval().to(config["device"])

    datamodule  = DataModule(config["datamodule"])
    #TODO Change to test
    datamodule.setup("predict")
    
    for dataset in datamodule.get_datasets("val"):
        logger.info(f"Testing on {dataset}")

        database_dataloader = DataLoader(
            dataset=dataset, num_workers=config["datamodule"]["num_workers"], batch_size=config["datamodule"]["batch_size"]
        )

        #TODO: Move from val to test datasets here... 
        #TODO: Split into database and query datasets... 

        logger.debug("Extracting database descriptors for evaluation/testing")
        #TODO: Check if passing the all_descriptors as key value ruins the speed up of creating the nump array 
        all_descriptors = np.empty((len(dataset), evaluation_params["descriptor_dimension"]), dtype="float32")
        all_descriptors = get_descriptors(model, all_descriptors, database_dataloader, config["device"])

    queries_descriptors = all_descriptors[dataset.num_references :]
    database_descriptors = all_descriptors[: dataset.num_references]

    if evaluation_params["save_descriptors"]:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(evaluation_params["descriptor_dimension"])
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(evaluation_params["recall_values"]))

    #TODO Calculate This better

    # # For each query, check if the predictions are correct
    # if evaluation_params["use_labels"]:
    #     positives_per_query = test_ds.get_positives()
    #     recalls = np.zeros(len(args.recall_values))
    #     for query_index, preds in enumerate(predictions):
    #         for i, n in enumerate(args.recall_values):
    #             if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
    #                 recalls[i:] += 1
    #                 break
    #     # Divide by num_queries and multiply by 100, so the recalls are in percentages
    # recalls = recalls / test_ds.num_queries * 100
    # recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    # logger.info(recalls_str)

    correct_at_k = np.zeros(len(evaluation_params["recall_values"]))
    gt = dataset.ground_truth
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(evaluation_params["recall_values"]):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
    correct_at_k = correct_at_k / len(predictions)
    d = {k:v for (k,v) in zip(evaluation_params["recall_values"], correct_at_k)}

    table = PrettyTable()
    table.field_names = ['K']+[str(k) for k in evaluation_params["recall_values"]]
    table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
    logger.info(table.get_string(title=f"Performance on {dataset.dataset_name}"))


    #Save visualizations of camera positions
    if evaluation_params["save_camera_positions"]:
        save_cameras(dataset, log_dir)
    # Save visualizations of predictions
    if evaluation_params["num_preds_to_save"] != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        save_preds(
            predictions[:, : evaluation_params["num_preds_to_save"] ], dataset, log_dir, evaluation_params["save_only_wrong_preds"], evaluation_params["use_labels"]
        )

if __name__ == "__main__":
    config = parse_args()
    
    evaluate(config)
