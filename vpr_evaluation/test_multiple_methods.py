import parser
import sys
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import visualizations
import vpr_models
from test_dataset import TestDataset
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def main(args):
    start_time = datetime.now()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)
    
    database_folder = np.load(args.database_folder).tolist()
    # q_folder = np.load(args.queries_folder).tolist()


    test_ds = TestDataset(
        database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    logger.info(f"Testing on {test_ds}")

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        for images, indices in tqdm(database_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
        for images, indices in tqdm(queries_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

    logger.debug(predictions)
    # # For each query, check if the predictions are correct
    # if args.use_labels:
    #     positives_per_query = test_ds.get_positives()
    #     recalls = np.zeros(len(args.recall_values))
    #     for query_index, preds in enumerate(predictions):
    #         for i, n in enumerate(args.recall_values):
    #             if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
    #                 recalls[i:] += 1
    #                 break

    #     # Divide by num_queries and multiply by 100, so the recalls are in percentages
    #     recalls = recalls / test_ds.num_queries * 100
    #     recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    #     logger.info(recalls_str)

    # groun_truths = np file 
    
    # correct_at_k = np.zeros(len(args.recall_values))
    # gt = np.load(args.ground_truth_folder, allow_pickle=True)
    # for q_idx, pred in enumerate(predictions):
    #     for i, n in enumerate(args.recall_values):
    #             # if in top N then also in top NN, where NN > N
    #             if np.any(np.in1d(pred[:n], gt[q_idx])):
    #                 correct_at_k[i:] += 1
    #                 break
    # correct_at_k = correct_at_k / len(predictions)
    # d = {k:v for (k,v) in zip(args.recall_values, correct_at_k)}


    # table = PrettyTable()   
    # table.field_names = ['K']+[str(k) for k in args.recall_values]
    # table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
    # logger.info(table.get_string(title=f"Performance "))

    # logger.info(correct_at_k)
    
    # plt.figure(figsize=(8, 6))
    # plt.plot(args.recall_values, correct_at_k, marker='o')
    # plt.xlabel('K')
    # plt.ylabel('Recall@K')
    # plt.title('Recall@K')
    # plt.grid(True)
    # plt.savefig(log_dir / 'recall_at_k_plot.png')


    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels
        )


methods = [ 
    # {
    #     "method" : "cosplace",
    #     "backbone": "ResNet18",
    #     "descriptors_dimension" : 512,
    #     "log_dir" : "results"
    # },
    # {
    #     "method" : "mixvpr",
    #     "backbone": "ResNet18",
    #     "descriptors_dimension" : 512,
    #     "log_dir" : "results"
    # },
    # {
    #     "method" : "salad",
    #     "backbone": "Dinov2",
    #     "descriptors_dimension" : 8448,
    #     "log_dir" : "results"
    # },

    # {
    #     "method" : "megaloc",
    #     "descriptors_dimension" : 10752,
    #     "backbone": "Dinov2",
    #     "log_dir" : "results"
    # },

    {
        "method" : "boq",
        "backbone": "Dinov2",
        "descriptors_dimension" : 12288,
        "image_size" : [322,322],
        "log_dir" : "images"
    },


    # {
    #     "method" : "boq",
    #     "backbone": "ResNet50",
    #     "descriptors_dimension" : 16384,
    #     "image_size" : [384,384],
    #     "log_dir" : "results"
    # }

    #     {
    #     "method" : "anyloc",
    #     "descriptors_dimension" : 49152,
    #     "backbone": "Dinov2",
    #     "log_dir" : "results"
    # },

    # {
    #     "method" : "netvlad",
    #     "descriptors_dimension" : 32768,
    #     "backbone": "ResNet18",
    #     "log_dir" : "results"
    # },  

]

if __name__ == "__main__":
    base_args = parser.parse_arguments()
    for method in methods:
        args = base_args
        args.method = method["method"]
        args.backbone = method["backbone"]
        args.num_preds_to_save = 3 
        args.log_dir = method["log_dir"] 
        args.use_labels = False
        args.image_size = method.get("image_size", args.image_size)
        print(method.get("image_size"))
        print(args.image_size)
        args.descriptors_dimension = method.get("descriptors_dimension", args.descriptors_dimension)


        # args.log_dir = method["log_dir"] + "/eiffel"
        # args.database_folder = "/home/rbeh9716/Desktop/UnderwaterVPR/data/val/Eiffel_subset/db_images.npy"
        # args.queries_folder = "/home/rbeh9716/Desktop/UnderwaterVPR/data/val/Eiffel_subset/q_images.npy"
        # args.ground_truth_folder = "/home/rbeh9716/Desktop/UnderwaterVPR/data/val/Eiffel_subset/gt.npy" 
    

        # main(args)

        # args.log_dir = method["log_dir"] + "/tofua"
        # args.database_folder = "/home/rbeh9716/Desktop/vpr_evaluation/datasets/Tofua/db_images.npy"
        # args.queries_folder = "/home/rbeh9716/Desktop/vpr_evaluation/datasets/Tofua/q_images.npy"
        # args.ground_truth_folder = "/home/rbeh9716/Desktop/vpr_evaluation/datasets/Tofua/gt.npy"
    
        # main(args)

        
        args.log_dir = method["log_dir"] + "/cross_view"
        args.database_folder = "/home/rbeh9716/Desktop/UnderwaterVPR/data/val/Tofua/db_images.npy"
        args.queries_folder = "/home/rbeh9716/Desktop/TongaProject/Datasets/data/GX010061/Frames"
        args.ground_truth_folder = "/home/rbeh9716/Desktop/UnderwaterVPR/data/val/Eiffel_subset/gt.npy" 
    

        main(args)

