import os
import random
import shutil
import pandas as pd

# Paths for main folder and destination for 5% selected files
main_folder = "/home/rbeh9716/Desktop/OpenVPRLab/data/train/tofua/Images(Copy)"
output_csv = 'folder_files.csv'
selected_files_csv = 'selected_files.csv'

# DataFrame to store information for CSV
folder_data = []
selected_files = []

# Loop through all subfolders
for subfolder in os.listdir(main_folder):
    if subfolder != "test":
        subfolder_path = os.path.join(main_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            # List all files in this subfolder
            

            files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
            
            # Add all files in the subfolder to the folder_data list
            for file in files:
                folder_data.append({'place_id': subfolder[6:], 'image_name': file})

# Create a DataFrame for all files in subfolders
df_folder_data = pd.DataFrame(folder_data)

df_folder_data = df_folder_data.sample(frac=1).reset_index(drop=True)
# Create a DataFrame for selected files
df_selected_files = pd.DataFrame(selected_files)

# Write DataFrames to CSV
df_folder_data.to_csv(output_csv, index=False)

print("Process completed!")
