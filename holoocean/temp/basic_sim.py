import holoocean
import matplotlib.pyplot as plt
import os 
import sys
from keyboard_controller import Controller
import numpy as np
from tqdm import tqdm
import json
import cv2

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

path = "configs/alex.json"
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
plot = ax.pcolormesh(T, R, z, cmap='copper', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
fig.canvas.draw()
fig.canvas.flush_events()


control  = Controller()
#### RUN SIMULATION

print("Generating Octrees...")
state = {}
config["scenario"]["agents"][0]["control_scheme"] = 0

with holoocean.make(scenario_cfg=config["scenario"], show_viewport=True) as env:
    while True:

        command = control.get_command(state)
        env.act("auv0", command)
        state = env.tick()

        if 'ImagingSonar' in state:
            s = state['ImagingSonar']
            plot.set_array(s.ravel())
            plt.savefig("output/sonar", bbox_inches='tight', pad_inches=0.5)

            fig.canvas.draw()
            fig.canvas.flush_events()

        if "RGBCamera" in state:
            print(state["RGBCamera"])

            cv2.imwrite("output/image.png", state["RGBCamera"])

# print("Finished Simulation!")
plt.ioff()
plt.show()
