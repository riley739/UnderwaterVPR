import pandas as pd
import numpy as np 
import glob 


def load_path_int_numpy(path, output):
    path = "/home/rbeh9716/Desktop/OpenVPRLab/data/val/tofua/q_images"

    files = {}

    for file in glob.glob(path + "/*.png"):
        img_name = file.split("/")[-1]
        files.append(file)
        # for index,row in df.iterrows():
        #     name = row["image_name"]


    array = np.array(files)

    np.save(output, array)


q_path = "/home/rbeh9716/Desktop/OpenVPRLab/data/val/tofua/tofua_q.npy"
db_path = "/home/rbeh9716/Desktop/OpenVPRLab/data/val/tofua/tofua_db.npy"


q_files = np.load(q_path).tolist()
db_files = np.load(db_path).tolist()

df = pd.read_csv("folder_files.csv")


files = []
for file in q_files:
    filename = file.split("/")[-1]
    result = df[df['image_name'] == filename]
    place_id  =  result.iloc[0]['place_id']
    q_matches = []

    for j,file in enumerate(db_files):
        filename = file.split("/")[-1]
        result = df[df['image_name'] == filename]
        db_place_id  =  result.iloc[0]['place_id']

        if db_place_id == place_id:
            q_matches.append(j)
        
        files.append(q_matches)

saved_array = np.array([np.array(xi) for xi in files], dtype="object")
print(saved_array)
np.save("/home/rbeh9716/Desktop/OpenVPRLab/data/val/tofua/tofua_gt.npy",saved_array)

print(np.load("/home/rbeh9716/Desktop/OpenVPRLab/data/val/tofua/tofua_gt.npy",allow_pickle=True).shape)
