from machinevisiontoolbox import Image, BagOfWords
from src.utils.files import PROJECT_ROOT
import os
import glob


def load_dataset(img_path):
    train_imgs = []
    test_imgs  = [] 
    for i,img in enumerate(glob.glob(img_path + "/*.jpg")):
        if i < 10:
            
            train_imgs.append(Image.Read(img))
        elif i < 20: 
            test_imgs.append(Image.Read(img))
        else:
            break

    return train_imgs, test_imgs


folder_path = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "test", "misc", "images" ))
train_images, test_imgs = load_dataset(folder_path)

print("LOADED ALL Images")

bow = BagOfWords(train_images)

print(bow.closest(train_images))
