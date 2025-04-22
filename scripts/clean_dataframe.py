import pandas as pd
import os
from tqdm import tqdm
df = pd.read_csv('data/val/Eiffel/Dataframes/eiffel.csv')

output_df =  pd.DataFrame(columns=['place_id', 'image_name']) 
rows = []
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_name = row["image_name"]
    if os.path.isfile(f"data/val/Eiffel/Images/{img_name}"):
        rows.append({"place_id": row["place_id"], "image_name": img_name})

output_df = pd.DataFrame(rows)

output_df.to_csv('data/val/Eiffel/Dataframes/Eiffel.csv', index=False)
