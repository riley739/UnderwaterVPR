import glob
import os 
from natsort import natsorted
from PIL import Image 
import random
import matplotlib.pyplot as plt
import numpy as np
import shutil


# path = "/home/rbeh9716/Desktop/OpenVPRLab/data/train/tofua/Images(Copy)"
# folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


directory = "/home/rbeh9716/Desktop/OpenVPRLab/data/train/tofua/Images(Copy)"

# Get all folders in the directory
folders = natsorted([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])

random_files = {}

for folder in folders:
    folder_path = os.path.join(directory, folder)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if files:  # Check if the folder has any files
        random_files[folder] = random.choice(files)

place_images = []
titles = []
# Print the random files selected from each folder
place_images.extend([Image.open(os.path.join(directory, folder, file)) for folder,file in random_files.items()])
titles.extend(folder for folder,file in random_files.items())
prev_place = ""
# for image in natsorted(glob.glob(directory + "/*.png")):    
#     img_name = image.split("/")[-1]
#     img = Image.open(image)


for i in range(len(place_images) // 25 + 1):
    if i < 8:
        continue
    
    fig, axes = plt.subplots(5, 5, figsize=(5 * 3, 10 * 3))  # Adjust figure size

    # # Plot the main image spanning all columns
    # ax_main = fig.add_subplot(gs[0, :])
    # ax_main.imshow(img, cmap="gray")
    # ax_main.set_title(img_name[-17:-9], fontsize=10)

    # ax_main.axis("off")

    # Plot the 10 smaller images
    for j,ax in enumerate(axes.flat):
        ax.imshow(place_images[min(len(place_images)-1,i*25 + j)], cmap="gray")
        ax.set_title(titles[min(len(place_images)-1, i*25 + j)], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.ion()
    # plt.show()
    plt.savefig(f"Places_{i*25}_{i*25+25}.jpg")

    plt.close()
    # correct = input(f"Enter [0-9] which place {img_name[-17:-9]} belongs too: \n")

    # try:
    #     val = int(correct)

    #     if i > 0:
    #         string = f"place_{i}{val}"
    #     else:
    #         string = f"place_{val}"

    #     shutil.move(img_name, directory + "/" + string + "/" + img_name)

    #     print(f"Moved file {img_name} to {string}")
    #     break
    # except ValueError:
    #     print("Moving to next one")

