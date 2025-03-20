import pickle
import matplotlib.pyplot as plt 
from numpy.linalg import norm
import numpy as np 
from datasets import load_dataset
import cv2 
from tqdm import tqdm

import glob 

with open("logs/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("logs/labels.pkl", "rb") as f:
    labels = pickle.load(f)

# imagenet = load_dataset(
#     'frgfm/imagenette',
#     'full_size',
#     split='train',
# )    
dataset = glob.glob("data/raw/images/*.jpg")
print(len(tfidf)    )
def load_image(n):
    return cv2.imread(dataset[n])

def get_base_accuracy():
    correct = 0 
    correct_images = [] 
    for n in tqdm(range(len(dataset))):
        a = tfidf[n]
        b = tfidf  # set search space to the full sample

        cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))

        top_k = 2
        idx = np.argsort(-cosine_similarity)[:top_k]

        label_img = int(idx[1])
        if labels[n] == labels[label_img]:
            correct += 1
            correct_images.append(n)
    
    print(correct/len(dataset))

# get_base_accuracy()


    
while val := int(input(f"Enter value between 0 - {len(dataset)}: ")):

    a = tfidf[val]
    b = tfidf  # set search space to the full sample

    cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))
    print(cosine_similarity)

    top_k = 5
    idx = np.argsort(-cosine_similarity)[:top_k]
    idx


    for i in idx:
        print(f"{i}: {round(cosine_similarity[i], 4)}")
        image = load_image(i)

        # image = np.array(imagenet[int(i)]['image'])

        # if len(image.shape) == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        # cv2.imshow("Input", image)
        # cv2.waitKey(0)
        plt.imshow(image, cmap='gray')
        plt.show()
