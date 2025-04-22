import holoocean
import numpy as np
import cv2
from math import atan2, degrees, radians
import matplotlib.pyplot as plt
from pynput import keyboard
from config import *
import csv
from PyQt5.QtCore import QLibraryInfo, QLibraryInfo
import os 

plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

pressed_keys = list()
force = 25

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.remove(key.char)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys:
        command[[4,7]] -= val
        command[[5,6]] += val

    if 'w' in keys:
        command[4:8] += val
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val

    return command

use_sonar = True

if use_sonar:
    config = config_final
else:
    config = config_wo_sonar

# Define waypoints
idx = 0
# locations = np.array([[*config["agents"][0]["location"]],
h = config["agents"][0]["rotation"][2]
p = config["agents"][0]["location"]
d_orig = p[2]
targets = [[0.0,1.8,-13.0],
             [13.5,17.5,-13.0],
             [10.1,1.5,-13.7]]
locations = [
    [-1.0,-8.0,d_orig],
    [-1.0,28.0,d_orig],
    [7.0,28.0,d_orig],
    [7.0,-8.0,d_orig],
    [15.0,-8.0,d_orig],
    [15.0,28.0,d_orig],
    [13.5,20,-13.7], # target 2
    [13.5,20,-11.0],
    [10.1,4,-13.0], # target 3
    [10.1,4,-10.3],
    [2.0,1.8,-13.0], # target 1
    [2.0,1.8,-10.3],
    [-1.0,-8.0,d_orig]
]
# locations = [
#     [15.0,28.0,d_orig],
#     [13.5,20,-13.7], # target 2
#     [13.5,20,-11.0],
#     [10.1,4,-13.0], # target 3
#     [10.1,4,-10.3],
#     [2.0,1.8,-13.0], # target 1
#     [2.0,1.8,-10.3],
#     [-1.0,-8.0,d_orig]
# ]

make_dataset = True
dataset_name = "clear_tracking"
# dataset_name = "clear_3targets"
dataset_path = f"datasets/{dataset_name}"

# cv2.namedWindow("Camera Output")

if use_sonar:
    sonar_config = config['agents'][0]['sensors'][-1]["configuration"]
    azi = sonar_config['Azimuth']
    minR = sonar_config['RangeMin']
    maxR = sonar_config['RangeMax']
    binsR = sonar_config['RangeBins']
    binsA = sonar_config['AzimuthBins']

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



# config = def
# Start simulation
with holoocean.make(scenario_cfg=config) as env:
# with holoocean.make(scenario_cfg=config, verbose=True) as env:
    # Draw waypoints
    for i, l in enumerate(locations[:-1]):
        env.draw_line(locations[i], locations[i+1], lifetime=0)
        # env.draw_point([l[0], l[1], -30], lifetime=0)

    locations = np.array(locations)
    print("Going to waypoint ", idx, locations[idx])

    if make_dataset:
        csv_file = open(f'{dataset_path}/dataset.csv', 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["time", "sensor", "id"])
        camera_id = 0
        state_id = 0
        sonar_low_id = 0
        sonar_high_id = 0
        dvl_id = 0

    speed_step_coeff = 1.0
    heading_step_coeff = 1.0
    heading_err_ratio = 1.0

    env.teleport_camera([5,5,5], [0, 90, 0])
    while True:
        if 'q' in pressed_keys:
            break
        command = parse_keys(pressed_keys, force)
        state = env.tick()

        if "RightCamera" in state:
            frame = state["RightCamera"]
            # cv2.imshow("Camera Output", frame[:, :, 0:3])
            # cv2.waitKey(1)
            if make_dataset:
                cv2.imwrite(f'{dataset_path}/camera/im_{camera_id:05}.jpg', frame[:, :, 0:3])
                writer.writerow([state['t'], "camera", camera_id])
                camera_id += 1
        if 'SonarLow' in state:
            s = state['SonarLow']
            plot.set_array(s.ravel())

            fig.canvas.draw()
            fig.canvas.flush_events()
            if make_dataset:
                np.savetxt(f'{dataset_path}/sonar_low/im_{sonar_low_id:05}.txt', s)
                writer.writerow([state['t'], "sonar_low", sonar_low_id])
                sonar_low_id += 1
        if 'SonarHigh' in state:
            s = state['SonarHigh']

            if make_dataset:
                np.savetxt(f'{dataset_path}/sonar_high/im_{sonar_high_id:05}.txt', s)
                writer.writerow([state['t'], "sonar_high", sonar_high_id])
                sonar_high_id += 1
        if 'DVL' in state:
            if make_dataset:
                np.savetxt(f'{dataset_path}/dvl/dvl_{dvl_id:05}.txt', state["DVL"])
                writer.writerow([state['t'], "dvl", dvl_id])
                dvl_id += 1
        if make_dataset:
            np.savetxt(f'{dataset_path}/state/st_{state_id:05}.txt', state["DynamicsSensor"])
            writer.writerow([state['t'], "state", state_id])
            state_id += 1
        p = state["DynamicsSensor"][6:9]
        o = state["DynamicsSensor"][15:18]
        h = o[2]
        h_d = degrees(atan2(locations[idx][1]-p[1], locations[idx][0]-p[0]))
        h_err = abs(h_d-h)
        if h_err > 50.0:
            hv_d = np.sign(h_d-h) * 50.0 * heading_step_coeff
            if heading_step_coeff < 1.0:
                heading_step_coeff += 0.1
        else:
            hv_d = h_d-h
        if h_err > 45.0:
            heading_err_ratio = (180-h_err)/(180-45)
        else:
            heading_err_ratio = 1.0
        dist_ = np.linalg.norm(locations[idx][0:2]-p[0:2])
        # dist_ = np.linalg.norm(locations[idx][0:3]-p[0:3])
        if dist_ > 1.0:
            v_d = [
                np.cos(radians(h))*speed_step_coeff*heading_err_ratio,
                np.sin(radians(h))*speed_step_coeff*heading_err_ratio,
                np.clip(locations[idx][2]-p[2], -0.5, 0.5)*speed_step_coeff
            ]
            if speed_step_coeff < 1.0:
                speed_step_coeff += 0.04
            # v_d = (locations[idx][0:3]-p[0:3])/np.linalg.norm(locations[idx][0:3]-p[0:3])
        else:
            v_d = [
                np.cos(radians(h))*dist_*heading_err_ratio,
                np.sin(radians(h))*dist_*heading_err_ratio,
                np.clip(locations[idx][2]-p[2], -0.5, 0.5)
            ]
            # v_d = locations[idx][0:3]-p[0:3]
        # env.agents["auv0"].set_physics_state(p, [0.0, 25.0, h], v_d, [0.0,0.0,hv_d])
        # if abs(h_d-h) > 20.0:
        #     env.agents["auv0"].set_physics_state(p, [0.0, 0.0, h], [0.0, 0.0, 0.0], [0.0,0.0,hv_d])
        # else:
        env.agents["auv0"].set_physics_state(p, [0.0, 0.0, h], v_d, [0.0,0.0,hv_d])
        # env.agents["auv0"].set_physics_state(p, [0.0, 0.0, h_d], v_d, [0.0,0.0,0.0])
        if np.linalg.norm(p[0:2]-locations[idx][0:2]) < 5.0e-1 and abs(p[2]-locations[idx][2]) < 0.33:
            speed_step_coeff = 0.0
            heading_step_coeff = 0.0
            idx = (idx+1) % len(locations)
            if idx == 0:
                break
            print("Going to waypoint ", idx,locations[idx])
print("Finished Simulation!")
if use_sonar:
    plt.ioff()
# cv2.destroyAllWindows()
