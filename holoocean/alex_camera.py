import holoocean, cv2
from src.utils import load_config

env = holoocean.make("Dam-HoveringCamera")
env.act('auv0', [10,10,10,10,0,0,0,0])
count = 0

while True:
    state = env.tick()
    if "LeftCamera" in state:
        pixels = state["LeftCamera"]
        print(count)
        count  += 1
        cv2.imwrite(f"camera/output_{count}.png", pixels[:, :, 0:3])
