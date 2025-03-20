

#Nodes = images  - come from images
#Node Features = Word Vectors from bovw - come from bovw
#edges = images from same place - come from datagram
#labels = places - come from dataframe

# Read in images
# Create bag of words

# Use the dataframes to create the links
 
# create the dataset using the youtube video

# Apply to the graph network in the youtube video 

## THIS IS CREATING A SINGLE GRAPH WHRE EACH NODE IS AN IMAGE defined by its tfidf vector 
## EDGES ARE BASESD ON WORD SIMILARITY
## LABELS ARE THE PLACE OF THE IMAGE
## EDGE FEATURES AARE NOT INTRODUCED YET
## COULD ALSO TRY EDGES ARE PURELY PLACE BASED? 


from torch_geometric.data import Dataset, Data
import torch
import pandas as pd 
from tqdm import tqdm
import cv2 
import os
import pickle 
import joblib
from datasets import load_dataset  
from collections import Counter
import numpy as np
from numpy.linalg import norm

class GraphDataset(Dataset):
    def __init__(self, root, transform = None, pre_transform = None):
        super().__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        return "images.csv"
    
    @property 
    def processed_file_names(self):
        return 'not_implemented.pt'
    

    def download(self):
        pass 

    def get_images(self):
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            path = os.path.abspath(f"{self.root}/raw/images/{row['key']}.jpg")
        
            yield cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
            
    def load_data(self):
        k, self.vocab = joblib.load("logs/bovw_vocab.pkl")


        with open("logs/frequency_vectors.pkl", "rb") as f:
            self.frequency_vectors = pickle.load(f)

        with open("logs/tfidf.pkl", "rb") as f:
            self.tfidf = pickle.load(f)

        with open("logs/labels.pkl", "rb") as f:
            self.labels = pickle.load(f)



    def process(self):
        # self.data = pd.read_csv(self.raw_paths[0])

        # print(self.data)

        self.load_data()

        # for i,image in enumerate(self.get_images()):

        
        nodes = self.get_node_features()
        edge_index = self.get_edges()
        label = self.get_labels()
        graph_data = Data(x=nodes, edge_index=edge_index, y=label)

        torch.save(graph_data, "logs/output.pt")
        
        exit()


    # 
    def get_node_features(self):
        # features = []
        # for i in range(0, len(self.tfidf)):
        #     features.append(self.tfidf[i])
        all_nodes = np.asarray(self.tfidf)

        return torch.tensor(all_nodes,dtype=torch.float)


    # def get_edges(self):

    #     def compare_images( img, comparison_img):
    #         img1 = Counter(img)
    #         img2 = Counter(comparison_img)
    #         common_words = img1 & img2
    #         return sum(common_words.values())
        
        
    #     edges = []
    #     for i,img in enumerate(self.frequency_vectors):
    #         for j,comparison_img in enumerate(self.frequency_vectors):
    #             if i != j:
    #                 similar = compare_images(self.frequency_vectors[i], comparison_img)

    #                 if similar > 50:
    #                     edges.append((i,j))

    #     edges = np.array(edges)
    #     edges = np.reshape(edges, (2, -1 ))
    #     return torch.tensor(edges, dtype=torch.long)

    def get_edges(self):
        edges = []
        b = self.tfidf

        for i in tqdm(range(0,len(self.tfidf))):
            a = self.tfidf[i]

            cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))

            idx = np.argsort(-cosine_similarity)[:5]

            for j in idx:
                if round(cosine_similarity[i], 4) > 0.8:
                    edges.append((i,j))
                else:
                    break

        edges = np.array(edges)
        edges = np.reshape(edges, (2, -1 ))
        return torch.tensor(edges, dtype=torch.long)
    

    # def get_edges(self):
    #     labels = [int(self.data[i]["label"]) for i in range(0,len(self.data))]

    #     edges = []
    #     for i in range(0,len(self.data)):
    #         label = self.data[i]["label"]    
    #         print(i)
    #         for j in range(0,len(self.data)):
    #             if i != j:
    #                if labels[i] == labels[j]:
    #                     edges.append((i,j))

    #     edges = np.array(edges)
    #     edges = np.reshape(edges, (2, -1 ))
    #     return torch.tensor(edges, dtype=torch.long)


    def get_labels(self):
        # labels = []
        # for i in range(0, len(self.data)):
        #     labels.append(self.data[i]['label'])

        labels = np.asarray(self.labels)
        return torch.tensor(labels, dtype=torch.int64)

if __name__ == "__main__":
    # img = cv2.imread("/home/rbeh9716/Desktop/UnderwaterVPR/gnn/data/raw/images/tsvRme3Cd1Xf4qXEV845qw.jpg", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    dataset = GraphDataset(root = 'data/')
    dataset.process()
