import pandas as pd 


df = pd.read_csv("/home/rbeh9716/Desktop/holoocean/holoocean.csv")

# Assuming you already have a DataFrame named df
df_shuffled = df.sample(frac=1, random_state=42)  # Shuffle the rows

# Calculate split index
split_idx = int(len(df_shuffled) * 0.66)

# Split into two DataFrames
df_66 = df_shuffled.iloc[:split_idx]
df_33 = df_shuffled.iloc[split_idx:]


df_66.to_csv('db_images.csv', index=False)
df_33.to_csv('q_images.csv', index=False)