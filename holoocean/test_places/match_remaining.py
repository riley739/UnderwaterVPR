import glob
import os 
from natsort import natsorted
from PIL import Image 
import random
import matplotlib.pyplot as plt
import numpy as np
import shutil
import cv2 
# path = "/home/rbeh9716/Desktop/OpenVPRLab/data/train/tofua/Images(Copy)"
# folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


directory = "/home/rbeh9716/Desktop/OpenVPRLab/data/train/tofua/Images(Copy)"


place_images = []
titles = []
place_images_dir = "/home/rbeh9716/Desktop/holoocean/test_places/images/"


# Print the random files selected from each folder
# print(natsorted([glob.glob(place_images_dir + "/*.jpg")]))
# for img in 
place_images.extend([cv2.imread(img) for img in natsorted(glob.glob(place_images_dir + "/*.jpg"))])

# for i,img in enumerate(place_images):
#     cv2.imshow(f"{i}", img)

# cv2.waitKey(0)
for img in glob.glob(directory + "/*.png"):
    image = cv2.imread(img)
    image = cv2.resize(image, (1080,720))
    # for i,matches in enumerate(place_images):
    #     # match = cv2.resize(matches, (1080,1080))
    #     cv2.imshow(f"{i}", matches)
    cv2.imshow(img, image)

    cv2.waitKey(0)
    img_name = img.split("/")[-1]

    correct = input(f"Enter which place {img_name[-17:-9]} belongs too: \n")

    try:
        val = int(correct)

        string = f"place_{val}"

        shutil.move(img, directory + "/" + string + "/" + img_name)

        print(f"Moved file {img_name} to {string}")
    except ValueError:
        print("Moving to next one")

