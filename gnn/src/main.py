# import numpy as np
# import cv2 
# from scipy.cluster.vq import kmeans, vq
# import glob 
# from numpy.linalg import norm
# import matplotlib.pyplot as plt
# import os
# from collections import Counter
# import pickle 
# from tqdm import tqdm
# import pandas as pd

# import joblib

# class BOVW():
#     def __init__(self, image_folder = ""):        
#         self.image_path = image_folder
#         self.k = 1000
#         self.iters = 1 

#         self.top_k = 5

#     def create_vocab(self):
#         print("STARTING")
#         self.load_dataset()
#         # print("LOADED DATASET")
        
#         self.extract_features()
#         print("EXTRACTED FEATURES")
#         self.cluster()
#         print("CLUSTERED")
#         self.get_sparse_frequency_vectors()
#         print("GET EDGES")
#         print("CALCULATED SPARNESS")
#         self.tf_idf()
#         print("FINISHED")

#         self.save_data()

#         print("SAVED IMAGES")

#     def save_data(self):    
#         joblib.dump((self.k, self.vocab), "logs/bovw_vocab.pkl", compress=3)

#         with open("logs/tfidf.pkl", "wb") as f:
#             pickle.dump(self.tfidf, f)
        
#         with open("logs/frequency_vectors.pkl", "wb") as f:
#             pickle.dump(self.frequency_vectors, f)

#         with open("logs/image_names.pkl", "wb") as f:
#             pickle.dump(self.img_names, f)


#         with open("logs/labels.pkl", "wb") as f:
#             pickle.dump(self.labels, f)
        
#     def extract_features(self, sample_size = 1000):
#         extractor = cv2.xfeatures2d.SIFT_create()

#         self.keypoints = [] 
#         self.descriptors = [] 

#         # self.get_images()
#         # for image in self.get_images():
#         for image in tqdm(self.bw_images):
#             img_keypoints, img_descriptors = extractor.detectAndCompute(image, None)

#             if img_descriptors is not None:
#                 self.descriptors.append(img_descriptors)
        
#         sample_idx = np.random.randint(0, self.num_images, 100).tolist()

#         descriptors_sample = []

#         for n in sample_idx:
#             descriptors_sample.append(np.array(self.descriptors[n]))

#         all_descriptors = []

#         for img_descriptors in descriptors_sample:
#             for descriptor in img_descriptors:
#                 all_descriptors.append(descriptor)

#         self.all_descriptors = np.stack(all_descriptors)

#         print(self.all_descriptors.shape)

#     def load_dataset(self):
#         training_images = []
#         img_names = []

#         self.data = pd.read_csv(f"{self.image_path}/images.csv")
#         img_names = [] 
#         img_labels = [] 
        
#         for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
#             path = os.path.abspath(f"{self.image_path}/images/{row['key']}.jpg")
#             img_names.append(row['key'])
#             img_labels.append(row['unique_cluster'])
#             training_images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            

#         self.img_names = img_names
#         self.bw_images = training_images
#         self.num_images = len(self.bw_images)
#         self.labels = img_labels
#         print(self.num_images)

#     def load_vocab(self):
#         k, vocab = joblib.load("bovw_vocab.pkl")

#     def cluster(self):
#         self.vocab,  variance = kmeans(self.all_descriptors, self.k, self.iters)

        
#         self.visual_words = []
#         for img_descriptors in self.descriptors:
#             if img_descriptors is not None:
#                 # for each image, map each descriptor to the nearest codebook entry
#                 img_visual_words, distance = vq(img_descriptors, self.vocab)
#                 self.visual_words.append(img_visual_words)
    
#     def get_sparse_frequency_vectors(self):
#         frequency_vectors = []
#         for img_visual_words in self.visual_words:
#             # create a frequency vector for each image
#             img_frequency_vector = np.zeros(self.k)
#             for word in img_visual_words:
#                 img_frequency_vector[word] += 1
#             frequency_vectors.append(img_frequency_vector)

#         # stack together in numpy array
#         self.frequency_vectors = np.stack(frequency_vectors)
#         self.frequency_vectors.shape

#     def tf_idf(self):
#         df = np.sum(self.frequency_vectors > 0, axis=0)
#         print(df)
#         print(self.frequency_vectors.shape)
#         idf = np.log(self.num_images/ df)

#         self.tfidf = self.frequency_vectors * idf

#     def search_img(self, img_num):
#         a = self.tfidf[img_num]
#         b = self.tfidf

#         cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))

#         idx = np.argsort(-cosine_similarity)[:self.top_k]

#         for i in idx:
#             print(f"{i}: {round(cosine_similarity[i], 4)}")
#             # plt.imshow(self.bw_images[i], cmap='gray')
#             # plt.show()

#     # def get_links(self):
#     #     self.links = []
#     #     for i,img in enumerate(self.frequency_vectors):
#     #         for j,comparison_img in enumerate(self.frequency_vectors):
#     #             if i != j:
#     #                 similar = self.compare_images(img, comparison_img)

#     #                 if similar > 50:
#     #                     self.links.append((i,j))
#     #                     print(f"Adding link between image {self.img_names[i]} and {self.img_names[j]}")

#     # def compare_images(self, img, comparison_img):
#     #     img1 = Counter(img)
#     #     img2 = Counter(comparison_img)
#     #     common_words = img1 & img2
#     #     print(common_words)
#     #     return sum(common_words.values())

#     def get_images(self):
#         self.data = pd.read_csv(f"{self.image_path}/images.csv")
#         img_names = [] 
#         self.num_images = self.data.shape[0]

#         for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
#             path = os.path.abspath(f"{self.image_path}/images/{row['key']}.jpg")
#             img_names.append(row['key'])
            
#             yield cv2.imread(path, cv2.IMREAD_GRAYSCALE)

#         self.img_names = img_names

#     # def get_images(self):
#     #     from datasets import load_dataset

#     #     imagenet = load_dataset(
#     #         'frgfm/imagenette',
#     #         'full_size',
#     #         split='train',
#     #     )        

#     #     # initialize list
#     #     images_training = []

#     #     for n in range(0,len(imagenet)):
#     #         # generate np arrays from the dataset images
#     #         images_training.append(np.array(imagenet[n]['image']))

#     #     bw_images = []
#     #     for img in images_training:
#     #         # if RGB, transform into grayscale
#     #         if len(img.shape) == 3:
#     #             bw_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#     #         else:
#     #             # if grayscale, do not transform
#     #             bw_images.append(img)
        
#     #     self.num_images = len(bw_images)
#     #     self.bw_images = bw_images

# def main():
#     folder_path = os.path.abspath("data/raw")
#     print(folder_path)
#     bovw = BOVW(folder_path)
#     bovw.create_vocab()
#     # bovw.search_img(1000)


# if __name__ == "__main__":
#     main()

import logging

logging.basicConfig(
    filename='out.log',  # Log file name
    level=logging.INFO,        # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)


logging.info("hello world ")
import time
time.sleep(10)


logging.info("done")
