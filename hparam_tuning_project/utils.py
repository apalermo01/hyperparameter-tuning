from hparam_tuning_project.training.callbacks import callback_registry
from typing import Dict
import yaml
import os


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


def update_cfg(old_cfg, new_cfg):
    for key in new_cfg:

        # check if this is a sub-dictionary
        if isinstance(new_cfg[key], dict):
            if key not in old_cfg:
                old_cfg[key] = {}
            old_cfg[key] = update_cfg(old_cfg[key], new_cfg[key])
        else:

            old_cfg[key] = new_cfg[key]
    return old_cfg


def load_cfg(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'base_cfg' in cfg:
        base_cfg = load_cfg(os.path.join(os.path.split(path)[0], cfg['base_cfg']))
        _ = cfg.pop('base_cfg')

        # looping through modified config options in cfg
        base_cfg = update_cfg(base_cfg, cfg)
        return base_cfg

    else:
        return cfg
