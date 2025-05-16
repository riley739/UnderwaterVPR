import pandas as pd 



base = pd.read_csv("test.csv")

places = pd.read_csv("/home/rbeh9716/Desktop/UnderwaterVPR/data/test/Rerank/query.csv")


merged_df = places.merge(base, on=["name","x","y","z"], how="left")


merged_df.to_csv("output.csv", index=False)
