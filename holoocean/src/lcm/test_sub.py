import lcm
from messages.PoseSensor import PoseSensor
from messages.RGBCamera import RGBCamera
import cv2 
import numpy as np 
import time

msg_count = 0
last_time = time.time()

def handler(channel, data):
    msg = PoseSensor.decode(data)
    print(f"Received: {msg.matrix}")

def rgb_handler(channel, data):
    start = time.time()
    msg = RGBCamera.decode(data)
    end = time.time()

    img_rgb = np.array(msg.image, dtype=np.uint8).reshape((msg.height, msg.width, 4))
    cv2.imshow("HoloOcean RGB Camera", img_rgb)
    cv2.waitKey(1)

    global msg_count, last_time
    msg_count += 1
    now = time.time()
    if now - last_time >= 1.0:
        print(f"Messages/sec: {msg_count}")
        msg_count = 0
        last_time = now


lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
# lc.subscribe("Pose", handler)
lc.subscribe("RGB", rgb_handler)

while True:
    lc.handle()
