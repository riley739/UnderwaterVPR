from datasets import load_dataset
import numpy as np
import cv2 

imagenet = load_dataset(
    'frgfm/imagenette',
    'full_size',
    split='train',
)        

images = [] 

# for n in range(0,len(imagenet)):
#     # generate np arrays from the dataset images
#     images.append(np.array(imagenet[n]['image']))


cv2.imshow('image',np.array(imagenet[8]['image']))
cv2.waitKey(0)
