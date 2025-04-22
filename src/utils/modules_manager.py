from src.core.framework import Framework
from src.utils.files import PROJECT_ROOT, get_path
import os
import importlib
import torch

def get_instance(module_name, class_name, params):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**params)


def get_backbone(config):
    return get_instance("src.backbones", config['model'], {'config': config['params']})


def get_aggregator(config):
    return get_instance("src.aggregators", config['method'], {'config': config['params']})

def get_loss_function(config):
    return get_instance("src.losses", config['class'], {'config': config['params']})

def create_model(config):
    backbone = get_backbone(config['backbone'])
    out_channels = backbone.out_channels # all backbones should have an out_channels attribute
    
    # most of the time, the aggregator needs to know the number of output channels of the backbone
    # that arguments is passed to the aggregator as a parameter `in_channels` for some aggregators
    #TODO: Hacky feel like theres a better way to do this..
    # Update aggregator's in_channels if necessary
    if 'in_channels' in config['aggregator']['params']:
        if config['aggregator']['params']['in_channels'] is None:
            config['aggregator']['params']['in_channels'] = out_channels

    # Create aggregator
    aggregator = get_aggregator(config['aggregator'])

    loss_function = get_loss_function(config['loss_function'])

    # Load model
    vpr_model = Framework(
        backbone=backbone,        
        loss_function=loss_function,
        aggregator=aggregator,
        config=config, # pass the config to the framework in order to save it
    )

    
    
    return vpr_model

def load_checkpoint(model, checkpoint_path):
    #Check if the path is a local path else load via url
    if  os.path.exists(checkpoint_path):
        # Load state_dict from the provided path
        state_dict_path = checkpoint_path
        if state_dict_path:
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                checkpoint_path,
                map_location=torch.device('cpu'),

            ), strict=False
        )    

#TODO: Update to be able to use paths outside of data but for now its fine...
#TODO: Since the train and val have differnet base datasets we should separate them out here
def get_dataset(name, config, dataset_type):
    

    path = os.path.join(PROJECT_ROOT, "data", dataset_type)
    path = get_path(name,path)

    params = {
        "name": name,
        "path": path,
        "config": config
    }
    try:
        dataset = get_instance("src.datasets", f"{name}{dataset_type.capitalize()}Dataset", params)
    except AttributeError:
        print("Dataset not found: Using default dataset")
        dataset = get_instance("src.datasets", f"Base{dataset_type.capitalize()}Dataset", params)
    return dataset 
