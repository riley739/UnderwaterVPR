import numpy as np
from util.camera_pose_visualizer import CameraPoseVisualizer
import json

if __name__ == '__main__':
    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 50])

    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera


    with open("poses.json","r") as f:
        poses = json.load(f)

    for image in poses["frames"]:

        se3 = np.array(image["pose"], dtype=np.float64)
        visualizer.extrinsic2pyramid(se3, int(image["place_id"])+1, 1)

    visualizer.show()