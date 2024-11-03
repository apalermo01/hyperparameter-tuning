from hparam_tuning_project.training.callbacks import callback_registry
from typing import Dict
import yaml
import os
import __main__
from datetime import datetime as dt
import boto3

# CONSTANTS
PATHS = {
    'dataset_path': '/home/alex/datasets/',
    'ap_local_path': '/home/alex/Documents/personal-projects/hyperparameter_tuning/',
    'linode_path': '/home/',
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


def _open_cfg(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'base_cfg' in cfg:
        base_cfg = _open_cfg(os.path.join(
            os.path.split(path)[0], cfg['base_cfg']))
        _ = cfg.pop('base_cfg')

        # looping through modified config options in cfg
        base_cfg = update_cfg(base_cfg, cfg)
        return base_cfg

    else:
        return cfg


def load_cfg(path, namespace_args=None):
    cfg = _open_cfg(path)

    if 'meta' not in cfg:
        cfg['meta'] = {}

    if hasattr(__main__, '__file__'):
        cfg['meta']['entry_script'] = __main__.__file__

    if namespace_args is not None:
        cfg['meta']['namespace_args'] = vars(namespace_args)

    cfg['meta']['start_time'] = str(dt.now())

    if 'flags' not in cfg or ('flags' in cfg and cfg['flags'] is None):
        cfg['flags'] = dict()

    cfg['data_cfg']['workdir'] = os.getcwd()
    return cfg


def move_results_to_linode_storage(results_path: str):
    pass
