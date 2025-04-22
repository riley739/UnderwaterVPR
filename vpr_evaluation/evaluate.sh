#!/bin/bash
export PATH="/home/rbeh9716/venv/python3.12/bin/python3.12:$PATH"

source ~/.bashrc
# Verify Python is in PATH
echo "Python version:"
python --version


python evaluate.py --method=boq --backbone=Dinov2 --descriptors_dimension=16384     --database_folder=/home/rbeh9716/Desktop/UnderwaterVPR/data/val/HoloOceanPlaces/db_images.npy --queries_folder=vscode-remote://ssh-remote%2B172.21.64.198/home/rbeh9716/Desktop/UnderwaterVPR/data/val/HoloOceanPlaces/q_images.npy     --no_labels --image_size 322 322 --num_preds_to_save 3     --log_dir holoocean --ground_truth_folder=vscode-remote://ssh-remote%2B172.21.64.198/home/rbeh9716/Desktop/UnderwaterVPR/data/val/HoloOceanPlaces/gt.npy