from datasets import load_dataset 
import pickle
import glob 
import pandas as pd 
from tqdm import tqdm 

def get_imagenet():
    imagenet = load_dataset(
        'frgfm/imagenette',
        'full_size',
        split='train',
    )    

    labels = [] 
    for n in range(len(imagenet)):
        labels.append(imagenet[n]["label"])

    with open("logs/labels.pkl", "wb") as f:
        pickle.dump(labels, f)





def get_csv():
    imgs = glob.glob("data/raw/images/*.jpg")

    print(len(imgs))
    data = pd.read_csv(f"data/raw/images.csv")
    labels = []

    for img in tqdm(imgs):
        imagename = img.split("/")[-1][:-4]
        labels.append(data.loc[data["key"] == imagename, "unique_cluster"].values[0])

    print(labels)
    print(len(labels))
    with open("logs/labels.pkl", "wb") as f:
        pickle.dump(labels, f)


get_csv()
