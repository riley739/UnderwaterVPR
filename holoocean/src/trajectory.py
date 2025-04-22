import numpy as np 
from scipy.spatial.transform import Rotation as R
from loguru import logger

class Trajectory:
    def __init__(self):
        self.starting_pose = None 
        self.start_world = None

    def transform_world_to_local(self, world_pose):
        
        T_start_pose = self.start_world @ world_pose

        # Extract new local position and orientation
        local_position = T_start_pose[:3, 3]
        local_rotation = R.from_matrix(T_start_pose[:3, :3]).as_quat()
    
        return local_position, local_rotation

    def save_pose(self, state):
        if self.starting_pose is None:
            self.starting_pose = state["PoseSensor"]
            self.start_world = np.linalg.inv(self.starting_pose)

        pose = state["PoseSensor"]
        logger.log("POSE", f"Pose: {pose}")
        
        
        # local_position, local_rotation = self.transform_world_to_local(pose)  

        # return local_position
