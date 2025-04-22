from src.datasets.val.base_dataset import BaseValDataset
import pandas as pd 
import os
class MSLSValDataset(BaseValDataset):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_df, self.q_df = self.load_dataframes()

    #TODO Figure out if problem if same name 
    def load_dataframes(self):

        database_cph = pd.read_csv(self.dataset_path  / "cph/database/postprocessed.csv")
        database_sf = pd.read_csv(self.dataset_path  / "sf/database/postprocessed.csv")
        query_cph = pd.read_csv(self.dataset_path  / "cph/query/postprocessed.csv")
        query_sf = pd.read_csv(self.dataset_path  / "sf/query/postprocessed.csv")

        d_df = pd.concat([database_cph, database_sf], ignore_index=True)
        q_df = pd.concat([query_cph, query_sf], ignore_index=True)  
        return d_df, q_df
    
    def get_pose(self, image_path, type):
        if type == "database":
            row = self.d_df.loc[self.d_df['key'] == os.path.basename(image_path).strip(".jpg")]["easting", "northing"]

            if not row.empty:
                pose = row.iloc[0].values
                return pose
        elif type == "query":
            row = self.q_df.loc[self.q_df['key'] == os.path.basename(image_path).strip(".jpg")]["easting", "northing"]

            if not row.empty:
                pose = row.iloc[0].values
                return pose

        return []


