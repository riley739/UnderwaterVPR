from src.utils.files import PROJECT_ROOT, get_path
import os
import importlib

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



#TODO: Update to be able to use paths outside of data but for now its fine...
#TODO: Since the train and val have differnet base datasets we should separate them out here
def get_dataset(name, config, dataset_type):
    

    path = os.path.join(PROJECT_ROOT, "data", dataset_type)
    path = get_path(name,path)

    print(path)

    params = {
        "name": name,
        "path": path,
        "config": config
    }
    
    dataset = get_instance("src.datasets", f"{name}{dataset_type.capitalize()}Dataset", params)

    return dataset 

    # if dataset_type == Types.TRAIN.value:
        
    
    # elif dataset_type == Types.TEST.value:
    
    
    # elif dataset_type == Types.VAL.value:
 