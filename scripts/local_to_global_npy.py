import numpy as np 


base_folder = "/home/rbeh9716/Desktop/UnderwaterVPR/data/val/MSLS/"
output_folder = "/home/rbeh9716/Desktop/UnderwaterVPR/scripts/"
q_folder = np.load(base_folder + "q_images.npy").tolist()

new_files = [] 
for path in q_folder:
    new_files.append(base_folder + path)


np.save(output_folder + "/q_images.npy", np.array(new_files))
