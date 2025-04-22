import pandas as pd 
import numpy as np 
import os 
from tqdm import tqdm
df = pd.read_csv("/home/rbeh9716/Desktop/UnderwaterVPR/data/train/Tofua/Dataframes/Tofua.csv")


db_images = np.load("/home/rbeh9716/Desktop/UnderwaterVPR/data/train/Tofua/db_images.npy")
q_images = np.load("/home/rbeh9716/Desktop/UnderwaterVPR/data/train/Tofua/q_images.npy")


def get_place_from_image_name(name):
    return df.loc[df['image_name'] == name]["place_id"].values[0]

gt = [] 
for img in tqdm(q_images):
    place = get_place_from_image_name(os.path.basename(img))
    matches = [] 
    for i,db_img in enumerate(db_images):
        if place == get_place_from_image_name(os.path.basename(db_img)):
            matches.append(i)

    gt.append(matches)

np.save("/home/rbeh9716/Desktop/UnderwaterVPR/data/train/Tofua/gt_test.npy", np.array(gt, dtype=object), allow_pickle=True)

print(len(q_images))
