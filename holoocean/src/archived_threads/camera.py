import cv2
import os
import glob
from loguru import logger
import threading 


class CameraThread():
    def __init__(self, log_dir = "data"):
        
        self.image = None
        self.save_image = False
        self.log_dir = log_dir
        self.img_count = 0
        self.img_dir = log_dir / "images"

        if os.path.isdir(self.img_dir):
            logger.warning("Image directory already exists, images will be appended to and may be overwritten!")
            self.img_count = len(glob.glob(self.img_dir + "/" + "*.png"))
        else:
            os.makedirs(self.img_dir)
            logger.info(f"Creating image directory: {self.img_dir}")
            
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.lock = threading.Lock()
        self.running = False


    def start(self):
        self.running = True
        self.display_thread.start()

    def stop(self):
        self.running = False
        self.display_thread.join()
        cv2.destroyAllWindows()

    def update_image(self,state):
    
        with self.lock:
            self.image = state["RGBCamera"]
            self.save_image = True
            self.img_count += 1
    
    def _save_image(self, image):
        img_name  = f"{self.img_dir}/{self.img_count:06d}.png"
        cv2.imwrite(img_name, image)


    def _display_loop(self):
        while self.running:
            with self.lock:
                img = self.image.copy() if self.image is not None else None
                if self.save_image:
                    self._save_image(img)
                    self.save_image = False
            if img is not None:
                cv2.imshow("Camera Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break







