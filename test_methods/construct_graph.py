

#Nodes = images  - come from images
#Node Features = Word Vectors from bovw - come from bovw
#edges = images from same place - come from datagram
#labels = places - come from dataframe

# Read in images
# Create bag of words

# Use the dataframes to create the links
 
# create the dataset using the youtube video

# Apply to the graph network in the youtube video 

from torch_geometric.data import Dataset, Data
import pandas as pd 

class GraphDataset(Dataset):
    def __init__(self, root, transform = None, pre_transform = None):
        super().__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        return "images.csv"
    
    @property 
    def procesed_file_names(self):
        return 'not_implemented.pt'
    

    def download(self):
        pass 


    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])

        for image in images:
            node = get_node
            edge_feats = get_edge_features #not needed atm
            edge_index = get_edges()
            label = get_label() 

        
            data = Data(x= node_features,
                        edge_index= edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        image_name = image_name
                        )


    def get_node_features(self):

        #  return torch.tensor(,dtype=torch.float)