import numpy as np
import cv2 
from scipy.cluster.vq import kmeans, vq
import glob 
from numpy.linalg import norm
import matplotlib.pyplot as plt
from datasets import load_dataset


class BOVW():
    def __init__(self, images = ""):
        self.images = load_dataset(
            'frgfm/imagenette',
            '160px',
            split='train',
            trust_remote_code=True,
        )

        

        self.k = 200
        self.iters = 1 

        self.top_k = 5

    def create_vocab(self):
        print("STARTING")
        self.load_dataset()
        print("LOADED DATASET")
        self.extract_features()
        print("EXTRACTED FEATURES")
        self.cluster()
        print("CLUSTERED")
        self.get_sparse_frequency_vectors()
        print("CALCULATED SPARNESS")
        self.tf_idf()
        print("FINISHED")

    def extract_features(self, sample_size = 1000):
        extractor = cv2.xfeatures2d.SIFT_create()

        self.keypoints = [] 
        self.descriptors = [] 

        for image in self.bw_images:
            img_keypoints, img_descriptors = extractor.detectAndCompute(image, None)

            if img_descriptors is not None:
                self.descriptors.append(img_descriptors)
        

        sample_idx = np.random.randint(0, self.num_images + 1, sample_size).tolist()

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

        # initialize list
        images_training = []

        for n in range(0,len(self.images)):
            # generate np arrays from the dataset images
            images_training.append(np.array(self.images[n]['image']))

        # convert images to grayscale
        self.bw_images = []
        for img in images_training:
            # if RGB, transform into grayscale
            if len(img.shape) == 3:
                self.bw_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            else:
                # if grayscale, do not transform
                self.bw_images.append(img)
        self.num_images = len(self.bw_images)

    # def get_all_images(self):
    #     for filepath in sorted(glob.glob(self.folder_path + "/*.png")):
    #         yield self.get_image(filepath)

    # def get_image(self, filepath):

    #     img = cv2.imread(filepath, 0)
    #     return img 

    def cluster(self):
        vocab,  variance = kmeans(self.all_descriptors, self.k, self.iters)

        self.visual_words = []
        for img_descriptors in self.descriptors:
            if img_descriptors is not None:
                # for each image, map each descriptor to the nearest codebook entry
                img_visual_words, distance = vq(img_descriptors, vocab)
                self.visual_words.append(img_visual_words)

    
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

    def tf_idf(self):
        df = np.sum(self.frequency_vectors > 0, axis=0)
        idf = np.log(self.num_images/ df)

        self.tfidf = self.frequency_vectors * idf


    def search_img(self, img_num):
        a = self.tfidf[img_num]
        b = self.tfidf

        cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))

        idx = np.argsort(-cosine_similarity)[:self.top_k]

        for i in idx:
            print(f"{i}: {round(cosine_similarity[i], 4)}")
            plt.imshow(self.bw_images[i], cmap='gray')
            plt.show()


def main():
    bovw = BOVW()
    bovw.create_vocab()
    bovw.search_img(6459)


if __name__ == "__main__":
    main()