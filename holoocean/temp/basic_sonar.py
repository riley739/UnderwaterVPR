import holoocean
import numpy as np
from tqdm import tqdm

command = np.array([0,0,0,0,-20,-20,-20,-20])
print("Building octrees...")
with holoocean.make("PierHarbor-HoveringImagingSonar") as env:
    for i in tqdm(range(1000)):
        env.act("auv0", command)
        state = env.tick()

print("Finished Simulation")
