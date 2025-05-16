import pandas as pd
import os
from tqdm import tqdm
import random
import numpy as np

base_path = "/home/rbeh9716/Desktop/UnderwaterVPR/data/val/holoocean/"
database = pd.read_csv("/home/rbeh9716/Desktop/UnderwaterVPR/data/val/holoocean/database/database_updated.csv")
query = pd.read_csv("/home/rbeh9716/Desktop/UnderwaterVPR/data/val/holoocean/query/query_updated.csv")

output_database = [] 
for index, row in tqdm(database.iterrows(), total=database.shape[0]):
    name = row["name"]
    output_database.append(base_path + f"database/images/{name}")

output_queries = []
ground_truth = []

for index, row in tqdm(query.iterrows(), total=query.shape[0]):
    place_id = row["place_id"]
    name = row["name"]
    
    output_queries.append(base_path + f"query/images/{name}")
    matches = [] 
    for index, db_row in database.iterrows():
        if place_id == db_row["place_id"]:
            matches.append(index)
    ground_truth.append(matches)



print(len(output_database))
print(len(output_queries))
print(len(ground_truth))


np.save("q_images.npy", np.array(output_queries))
np.save("db_images.npy", np.array(output_database))
np.save("gt.npy", np.array(ground_truth, dtype=object), allow_pickle=True)

