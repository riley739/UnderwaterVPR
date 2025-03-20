import numpy as np
import cv2 
from scipy.cluster.vq import kmeans, vq
import glob 
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os
from collections import Counter
import pickle 
from tqdm import tqdm
import pandas as pd
from PIL import Image

import joblib

import logging

# logging.basicConfig(
#     filename='out.log',  # Log file name
#     level=print,        # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#     format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
# )



class BOVW():
    def __init__(self, image_folder = ""):        
        self.image_path = image_folder
        self.k = 200
        self.iters = 1 

        self.top_k = 5

    def create_vocab(self):
        # print("STARTING")
        # self.load_dataset_imagenet()
        print("Extracting Features")
        self.extract_features()
        print("EXTRACTED FEATURES")
        self.cluster()
        print("CLUSTERED")
        self.get_sparse_frequency_vectors()
        print("GET EDGES")
        print("CALCULATED SPARNESS")
        self.tf_idf()
        print("FINISHED")

        self.save_data()

        print("SAVED IMAGES")

    def save_data(self):    
        joblib.dump((self.k, self.vocab), "logs/bovw_vocab.pkl", compress=3)

        with open("logs/tfidf.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)
        
        with open("logs/frequency_vectors.pkl", "wb") as f:
            pickle.dump(self.frequency_vectors, f)


        with open("logs/labels.pkl", "wb") as f:
            pickle.dump(self.labels, f)

        with open("logs/image_names.pkl", "wb") as f:
            pickle.dump(self.img_names, f)
        
    def extract_features(self, sample_size = 1000):

        # from datasets import load_dataset

        # imagenet = load_dataset(
        #     'frgfm/imagenette',
        #     'full_size',
        #     split='train',
        # )      
        import glob 

        imgs = glob.glob("data/raw/images/*.jpg")
        self.num_images = len(imgs)
        extractor = cv2.xfeatures2d.SIFT_create()

        self.keypoints = [] 
        self.descriptors = [] 
        self.labels = []
        # for n in tqdm(range(0,self.num_images), total=self.num_images):
        for img in tqdm(imgs, total=self.num_images):
            # image = np.array(imagenet[n]['image'])

            # if len(image.shape) == 3:
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
            # self.labels.append(imagenet[n]["label"])
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            img_keypoints, img_descriptors = extractor.detectAndCompute(image, None)

            if img_descriptors is not None:
                self.keypoints.append(img_keypoints)
                self.descriptors.append(img_descriptors)
        
        sample_idx = np.random.randint(0, self.num_images, sample_size).tolist()

        print("Done Extracting now sampling")
        descriptors_sample = []

        for n in sample_idx:
            descriptors_sample.append(np.array(self.descriptors[n]))

        all_descriptors = []

        for img_descriptors in descriptors_sample:
            for descriptor in img_descriptors:
                all_descriptors.append(descriptor)

        self.all_descriptors = np.stack(all_descriptors)

        print(self.all_descriptors.shape)

    def load_dataset(self):
        training_images = []
        img_names = []

        self.data = pd.read_csv(f"{self.image_path}/images.csv")
        img_names = [] 
        img_labels = [] 
        
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            path = os.path.abspath(f"{self.image_path}/images/{row['key']}.jpg")
            img_names.append(row['key'])
            img_labels.append(row['unique_cluster'])
            training_images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            

        self.img_names = img_names
        self.bw_images = training_images
        self.num_images = len(self.bw_images)
        self.labels = img_labels
        print(self.num_images)

    def load_dataset_imagenet(self):
        from datasets import load_dataset

        imagenet = load_dataset(
            'frgfm/imagenette',
            'full_size',
            split='train',
        )        
 
        self.num_images = len(imagenet)

        def load_image():
            for n in range(0,len(imagenet)):
                # generate np arrays from the dataset images
                img = np.array(imagenet[n]['image'])

                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                yield img

        self.bw_images = load_image()

    def load_vocab(self):
        k, vocab = joblib.load("bovw_vocab.pkl")

    def cluster(self):
        self.vocab,  variance = kmeans(self.all_descriptors, self.k, self.iters)

        
        self.visual_words = []
        for img_descriptors in self.descriptors:
            if img_descriptors is not None:
                # for each image, map each descriptor to the nearest codebook entry
                img_visual_words, distance = vq(img_descriptors, self.vocab)
                self.visual_words.append(img_visual_words)

        print(self.visual_words[0][:5], len(self.visual_words[0]))


    def get_sparse_frequency_vectors(self):
        frequency_vectors = []
        for img_visual_words in self.visual_words:
            # create a frequency vector for each image
            img_frequency_vector = np.zeros(self.k)
            for word in img_visual_words:
                img_frequency_vector[word] += 1
            frequency_vectors.append(img_frequency_vector)

        # stack together in numpy array
        self.frequency_vectors = np.stack(frequency_vectors)
        self.frequency_vectors.shape

        for i in [84,  22,  45, 172]:
            print(f"{i}: {self.frequency_vectors[0][i]}")

        print(self.frequency_vectors[0][:20])

        #plt.bar(list(range(self.k)), self.frequency_vectors[0])
        #plt.show()

    def tf_idf(self):
        df = np.sum(self.frequency_vectors > 0, axis=0)

        print(df.shape)
        print(df[:5])
        idf = np.log(self.num_images/ df)

        print(idf)
        print(idf.shape)

        self.tfidf = self.frequency_vectors * idf

        print(self.tfidf.shape)
        print(self.tfidf[0][:5])

        #plt.bar(list(range(self.k)), self.tfidf[0])
        #plt.show()

    def search_img(self, img_num):
        a = self.tfidf[img_num]
        b = self.tfidf

        cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))

        idx = np.argsort(-cosine_similarity)[:self.top_k]

        for i in idx:
            print(f"{i}: {round(cosine_similarity[i], 4)}")
            # #plt.imshow(self.bw_images[i], cmap='gray')
            # #plt.show()

   

def main():
    # folder_path = os.path.abspath("data/raw")
    # print(folder_path)
    bovw = BOVW("")
    bovw.create_vocab()


if __name__ == "__main__":
    main()


