import holoocean
import numpy as np
import json
from loguru import logger

from .control.controller import Controller
from .camera import CameraReader
from .utils import save_pose
from .trajectory import Trajectory
from .sonar import SonarReader

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def simulation(config):    
    logger.info("Loading Simulation")
    
    #TODO: This will need to be updated for multiple agents 
    controller = Controller(config["scenario"]["agents"][0])
    camera = CameraReader(config["log_dir"])
    trajectory = Trajectory()
    sonar = SonarReader(config)

    env = holoocean.make(scenario_cfg=config["scenario"], show_viewport=config["display_window"])
    logger.info("Simulation Running")


    count = 0 
    state = {} 
    while True:
        command = controller.get_command(state)
        env.act("auv0", command)
        state = env.tick()

        
        count += 1

        if "RGBCamera" in state:
            camera.save_image(state)
            

        if "ImagingSonar" in state:
            sonar.save_data(state)
        
        if "PoseSensor" in state:
            trajectory.save_pose(state)
