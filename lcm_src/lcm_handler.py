import lcm
import numpy as np
from PIL import Image
import threading

from lcm_src.messages.RGBCamera import RGBCamera

class LCMImageReceiver:
    def __init__(self, topic="RGB"):
        self.lcm_instance = lcm.LCM()
        self.topic = topic
        self.latest_image = None
        self.lock = threading.Lock()

        self.lcm_instance.subscribe(self.topic, self._handler)

    def _handler(self, channel, msg):
        decoded = RGBCamera.decode(msg)
        try:
            img = np.array(decoded.image, dtype=np.uint8).reshape((decoded.height, decoded.width, decoded.channels))
            img = Image.fromarray(img).convert("RGB")

            with self.lock:
                self.latest_image = img

        except Exception as e:
            print(f"[LCMImageReceiver] Error decoding image: {e}")

    def get_latest_image(self):
        with self.lock:
            img = self.latest_image
            self.latest_image = None
        return img

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
