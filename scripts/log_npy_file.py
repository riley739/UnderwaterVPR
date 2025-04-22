
import numpy as np 

db_images = np.load("/home/rbeh9716/Desktop/UnderwaterVPR/data/val/HoloOceanPlaces/db_images.npy")



with open("scripts/log.txt", "w+") as f:
    for img in db_images:
        f.write(img)
        f.write("\n")

print(len(db_images))
