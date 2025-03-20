import json

def load_config(config_path='model_config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)



# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse
import yaml
from typing import Dict, Any


def open_config(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='VPR Framework Training and Evaluation')

    # General arguments
    parser.add_argument('--config', type=str, default="configs/default.yaml", help='Path to the JSON configuration file')
    parser.add_argument('--seed', type=int, default = 42, help='Random seed for reproducibility')
    parser.add_argument('--silent', action='store_true', default=False, help='Disable verbose output')
    parser.add_argument('--compile', action='store_true', default=False, help='Compile the model using torch.compile()')
    parser.add_argument('--dev', action='store_true', default=False, help='Enable fast development run')
    parser.add_argument('--display_theme', type=str, default="default", help='Theme for the console display')
    parser.add_argument('--checkpoint', type=str, help='Theme for the console display')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for training and evaluation')
    
    # Datamodule arguments
    parser.add_argument('--train_set', type=str, help='Name of the training dataset')
    parser.add_argument('--val_sets', nargs='+', help='Names of the validation datasets')
    parser.add_argument('--train_image_size', type=int, nargs=2, help='Training image size (height width)')
    parser.add_argument('--val_image_size', type=int, nargs=2, help='Validation image size (height width). Dafault is None (same as training size)')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--img_per_place', type=int, help='Number of images per place')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')

    # Model arguments
    parser.add_argument('--backbone', type=str, help='Backbone model name')
    parser.add_argument('--aggregator', type=str, help='Aggregator model name')
    parser.add_argument('--loss_function', type=str, help='Loss function name')

    # Trainer arguments
    parser.add_argument('--optimizer', type=str, help='Optimizer name')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    parser.add_argument('--warmup', type=int, help='Number of warmup steps')
    parser.add_argument('--milestones', nargs='+', type=int, help='Milestones for learning rate scheduler')
    parser.add_argument('--lr_mult', type=float, help='Learning rate multiplier for scheduler')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')

    args = parser.parse_args()

    # If a config file is provided, load it
    config = open_config(args.config)

    # Update config with command-line arguments and default values
    config = update_config_with_args_and_defaults(config, args)

    return config

def update_config_with_args_and_defaults(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update the configuration dictionary with command-line arguments and default values.
    Priority: Command-line args > Config file values > Default values
    """

    # Helper function to update nested dictionaries
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    # Update with command-line arguments if provided
    arg_dict = vars(args)
    
    # Update datamodule config
    if arg_dict['train_set'] is not None:
        config['datamodule']['train_set_name'] = arg_dict['train_set']
    if arg_dict['val_sets'] is not None:
        config['datamodule']['val_set_names'] = arg_dict['val_sets']
    if arg_dict['train_image_size'] is not None:
        config['datamodule']['train_image_size'] = arg_dict['train_image_size']
    if arg_dict['val_image_size'] is not None:
        config['datamodule']['val_image_size'] = arg_dict['val_image_size']
    if arg_dict['batch_size'] is not None:
        config['datamodule']['batch_size'] = arg_dict['batch_size']
    if arg_dict['img_per_place'] is not None:
        config['datamodule']['img_per_place'] = arg_dict['img_per_place']
    if arg_dict['num_workers'] is not None:
        config['datamodule']['num_workers'] = arg_dict['num_workers']

    # Update model config
    if arg_dict['backbone'] is not None:
        config['backbone']['class'] = arg_dict['backbone']
    if arg_dict['aggregator'] is not None:
        config['aggregator']['class'] = arg_dict['aggregator']
    if arg_dict['loss_function'] is not None:
        config['loss_function']['class'] = arg_dict['loss_function']

    # Update trainer config
    if arg_dict['optimizer'] is not None:
        config['trainer']['optimizer'] = arg_dict['optimizer']
    if arg_dict['lr'] is not None:
        config['trainer']['lr'] = arg_dict['lr']
    if arg_dict['wd'] is not None:
        config['trainer']['wd'] = arg_dict['wd']
    if arg_dict['warmup'] is not None:
        config['trainer']['warmup'] = arg_dict['warmup']
    if arg_dict['milestones'] is not None:
        config['trainer']['milestones'] = arg_dict['milestones']
    if arg_dict['lr_mult'] is not None:
        config['trainer']['lr_mult'] = arg_dict['lr_mult']
    if arg_dict['max_epochs'] is not None:
        config['trainer']['max_epochs'] = arg_dict['max_epochs']

    # Update other general config
    if arg_dict['seed'] is not None:
        config['seed'] = arg_dict['seed']
    if arg_dict['silent'] is not None:
        config['silent'] = arg_dict['silent']
    if arg_dict['compile'] is not None:
        config['compile'] = arg_dict['compile']
    if arg_dict['display_theme'] is not None:
        config['display_theme'] = arg_dict['display_theme']
    if arg_dict['dev'] is not None:
        config['dev'] = arg_dict['dev']
    if arg_dict["checkpoint"] is not None:
        config["checkpoint"] = arg_dict["checkpoint"]
    if arg_dict["device"] is not None:
        config["device"] = arg_dict["device"]
    
    return config

if __name__ == "__main__":
    config = parse_args()
