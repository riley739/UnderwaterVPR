import os, sys
import argparse
import numpy as np

from src.visualizer import CameraVisualizer
from src.loader import load_quick
from src.utils import load_image, rescale_cameras, recenter_cameras

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--format', default='quick', choices=['quick', 'nerf', 'colmap'])
parser.add_argument('--type', default=None, choices=[None, 'sph', 'xyz', 'elu', 'c2w', 'w2c'])
parser.add_argument('--no_images', action='store_true')
parser.add_argument('--mesh_path', type=str, default=None)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--scene_size', type=int, default=100)
parser.add_argument('--y_up', action='store_true')
parser.add_argument('--recenter', action='store_true')
parser.add_argument('--rescale', type=float, default=None)

args = parser.parse_args()

root_path = args.root

poses = []
legends = []
colors = []
images = None

if args.format == 'quick':
    poses, image_paths, colors = load_quick(root_path)


if args.recenter:
    poses = recenter_cameras(poses)

if args.rescale is not None:
    poses = rescale_cameras(poses, args.rescale)

if args.y_up:
    for i in range(0, len(poses)):
        poses[i] = poses[i][[0, 2, 1, 3]]
        poses[i][1, :] *= -1
    

viz = CameraVisualizer(poses, image_paths, colors)
fig = viz.update_figure(args.scene_size, base_radius=1, zoom_scale=1, show_grid=True, show_ticklabels=False, show_background=True, y_up=args.y_up)

fig.show()
