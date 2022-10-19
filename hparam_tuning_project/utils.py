from hparam_tuning_project.training.callbacks import callback_registry
from typing import Dict


### CONSTANTS
PATHS = {
    'dataset_path': '/home/alex/datasets/'
}


def initialize_callbacks(cfg):
    callbacks = []
    for call in cfg:
        name = list(call.keys())[0]
        args = call[name]
        if args is None:
            args = {}
        callbacks.append(callback_registry[name](**args))
    return callbacks
