import os
import numpy as np
import json

from .utils import elu_to_c2w, spherical_to_cartesian, load_image, qvec_to_rotmat, rotmat


def load_quick(root_path):

    poses = []
    legends = []
    colors = []
    image_paths = []


    pose_path = os.path.join(root_path, 'poses.json')
    print(f'Load poses from {pose_path}')
    with open(pose_path, 'r') as fin:
        jdata = json.load(fin)
    frame_list = jdata['frames']
    

    for idx, frame in enumerate(frame_list):

        fid = idx

        c2w = np.array(frame['pose'])
        if c2w.shape[0] == 3:
            c2w = np.concatenate([c2w, np.zeros((1, 4))], axis=0)
            c2w[-1, -1] = 1


        poses.append(c2w)
        colors.append(int(frame["place_id"]))
        image_paths.append(frame['image_name']) 

    return poses, image_paths, colors


