import holoocean
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


path = "configs/sonar_test.json"
#### GET SONAR CONFIG
config = load_config(path)

configuration = config["scenario"]['agents'][0]['sensors'][-1]["configuration"]
azi = configuration['Azimuth']
minR = configuration['RangeMin']
maxR = configuration['RangeMax']
binsR = configuration['RangeBins']
binsA = configuration['AzimuthBins']

#### GET PLOT READY
plt.ion()
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8,5))
ax.set_theta_zero_location("N")
ax.set_thetamin(-azi/2)
ax.set_thetamax(azi/2)

theta = np.linspace(-azi/2, azi/2, binsA)*np.pi/180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)
z = np.zeros_like(T)

plt.grid(False)
plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
fig.canvas.draw()
fig.canvas.flush_events()

#### RUN SIMULATION
command = np.array([0,0,0,0,-20,-20,-20,-20])
print("Generating Octrees...")
with holoocean.make(scenario_cfg=config["scenario"], show_viewport=True) as env:
    for i in tqdm(range(1000)):
        env.act("auv0", command)
        state = env.tick()

        if 'ImagingSonar' in state:
            s = state['ImagingSonar']
            plot.set_array(s.ravel())

            fig.canvas.draw()
            fig.canvas.flush_events()

print("Finished Simulation!")
plt.ioff()
plt.show()
