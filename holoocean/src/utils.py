from loguru import logger
import os 
import json 

#TODO: Make the pose string and ensure its correct
def pose_string(pose):
    print(pose)
    exit()
#     """
#     Convert a 4x4 pose matrix to a string representation.
#     """
#     pose_str = f"[[{pose[0, 0]:.6f},{pose[0, 1]:.6f},{pose[0, 2]:.6f},{pose[0, 3]:.6f}],"
#     pose_str += f"[{pose[1, 0]:.6f},{pose[1, 1]:.6f},{pose[1, 2]:.6f},{pose[1, 3]:.6f}],"
#     pose_str += f"[{pose[2, 0]:.6f},{pose[2, 1]:.6f},{pose[2, 2]:.6f},{pose[2, 3]:.6f}],"
#     pose_str += f"[{pose[3, 0]:.6f},{pose[3, 1]:.6f},{pose[3, 2]:.6f},{pose[3, 3]:.6f}]]"
#     return pose_str

def save_pose(state):
    #TODO Move this elsewhere and make it much faster
    pose = state["PoseSensor"]  
    logger.log("POSE", f"Pose: {pose}")


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
