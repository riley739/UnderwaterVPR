import pandas as pd
import os
from tqdm import tqdm
import random
import numpy as np

base_folder =  os.path.abspath("data/train/HoloOceanPlaces")
query_folder = os.path.abspath("data/val/HoloOceanPlaces")
print(base_folder)
df = pd.read_csv(base_folder + '/Dataframes/HoloOceanPlaces.csv')

places = {}
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    place_id = row["place_id"]
    image_name = row["image_name"]

    if place_id not in places:
        places[place_id] = [image_name]
    else:
        places[place_id].append(image_name)

output_queries = []
output_database = [] 
ground_truth = [] 

for place in tqdm(places.values()):
    random.shuffle(place)

    barrier = 2* len(place) // 3  
    queries = place[barrier:]
    db = place[:barrier]   

    for query in queries:
        output_queries.append(base_folder + f"/Images/{query}")
        matches = [] 
        for i in range(len(db)):
            matches.append(len(output_database) + i)
        ground_truth.append(matches)

    for database in db:
        output_database.append(base_folder + f"/Images/{database}")


print(len(output_database))
print(len(output_queries))
print(len(ground_truth))
print(df.shape)


np.save(query_folder + "/q_images.npy", np.array(output_queries))
np.save(query_folder + "/db_images.npy", np.array(output_database))
np.save(query_folder + "/gt.npy", np.array(ground_truth, dtype=object), allow_pickle=True)

