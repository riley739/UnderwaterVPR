import cv2
import os
import glob
from loguru import logger

#TODO Move to thread if slow
class CameraReader():
    def __init__(self, log_dir = "data"):

        self.log_dir = log_dir
        self.img_count = 0
        self.img_dir = log_dir / "images"

        if os.path.isdir(self.img_dir):
            logger.warning("Image directory already exists, images will be appended to and may be overwritten!")
            self.img_count = len(glob.glob(self.img_dir + "/" + "*.png"))
        else:
            os.makedirs(self.img_dir)
            logger.info(f"Creating image directory: {self.img_dir}")

    def save_image(self, state):

        img_name  = f"{self.img_dir}/{self.img_count:06d}.png"
        image = state["RGBCamera"]
        cv2.imwrite(img_name, image)
        self.img_count += 1


        logger.log("CAMERA", f"Image: {img_name}\nPose: {state['PoseSensor']}")







