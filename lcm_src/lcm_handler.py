import lcm
import numpy as np
from PIL import Image
import threading

from lcm_src.messages.RGBCamera import RGBCamera
from lcm_src.messages.PoseSensor import PoseSensor  # Adjust this to your actual message name

class LCMImageReceiver:
    def __init__(self, image_topic="RGB", pose_topic="Pose"):
        self.lcm_instance = lcm.LCM()
        self.image_topic = image_topic
        self.pose_topic = pose_topic

        self.latest_image = None
        self.latest_pose = None
        self.lock = threading.Lock()

        self.lcm_instance.subscribe(self.image_topic, self._image_handler)
        self.lcm_instance.subscribe(self.pose_topic, self._pose_handler)

    def _image_handler(self, channel, msg):
        decoded = RGBCamera.decode(msg)
        try:
            img = np.array(decoded.image, dtype=np.uint8).reshape((decoded.height, decoded.width, decoded.channels))
            img = img[:, :, :3]      # Drop alpha
            img = img[..., ::-1]     # Convert RGBA → BGR → RGB
            img = Image.fromarray(img)

            with self.lock:
                self.latest_image = img

        except Exception as e:
            print(f"[LCMImageReceiver] Error decoding image: {e}")

    def _pose_handler(self, channel, msg):
        try:
            decoded = PoseSensor.decode(msg).matrix
            with self.lock:
                self.latest_pose = [decoded[0][-1], decoded[1][-1], decoded[2][-1]]
        except Exception as e:
            print(f"[LCMImageReceiver] Error decoding pose: {e}")

    def get_latest_image(self):
        with self.lock:
            img = self.latest_image
            self.latest_image = None
        return img

    def get_latest_pose(self):
        with self.lock:
            return self.latest_pose  # Do not clear pose unless you want one-shot access

    def handle_once(self, timeout_ms=None):
        if timeout_ms is not None:
            self.lcm_instance.handle_timeout(timeout_ms)
        else:
            self.lcm_instance.handle()

    def start_background_thread(self):
        thread = threading.Thread(target=self._loop_forever, daemon=True)
        thread.start()

    def _loop_forever(self):
        while True:
            self.lcm_instance.handle()
