import json
import os
import matplotlib.pyplot as plt
from PIL import Image 
import math 
import shutil

# Load JSON file
json_file = "clustered_se3_poses_2.json"  # Change this to your actual JSON filename
with open(json_file, "r") as file:
    data = json.load(file)

# Set image directory (change this if needed)
image_directory = "/home/rbeh9716/Desktop/OpenVPRLab/data/train/tofua/Images(Copy)"  # Set the folder where images are stored

# Collect all image file paths from JSON
for key in data:
    image_paths = []
    titles = []
    print(f"Looking at place {key}")
    image_paths.extend([os.path.join(image_directory, img) for img in data[key]])
    titles.extend([img[-17:-9] for  img in data[key]])
    images = [Image.open(img).resize((250,250)) for img in image_paths]


    cols = 5
    rows = math.ceil(len(images) / cols)

    # Display all images side by side
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust figure size as needed
    
    if rows == 1:
        axes = [axes]  # Make it a list for iteration
    if cols == 1:
        axes = [[ax] for ax in axes]  # Ensure 2D structure

    # Flatten axes for easy iteration
    axes = [ax for row in axes for ax in row]  

    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Hide any extra empty subplots (if the last row is incomplete)
    for j in range(i + 1, len(axes)):
      axes[j].axis("off")

    new_dir = image_directory + f"/place_{key}"

            

    # plt.tight_layout()
    plt.ion()  
    plt.show()

    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)

    os.mkdir(new_dir)
    for img in image_paths:
        correct = input(f"Enter if {img[-17:-9]} is correct: \n")

        if correct == "n":
            print("Did not add")
        else:
            shutil.move(img, new_dir + "/" + img.split("/")[-1])
            print("Copied")
    plt.close()