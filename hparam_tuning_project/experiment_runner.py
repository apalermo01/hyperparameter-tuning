"""Functionality to run one iteration of the experiment"""

from typing import Dict
import yaml
from hparam_tuning_project.training.trainer import Trainer
from hparam_tuning_project.data.datasets import PytorchDataset
import pytorch_lightning as pl

config_path = "/home/alex/Documents/personal-projects/hyperparameter-tuning/training_configs/"
config_name = "simple_training.yaml"


def load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # print(cfg)
    return cfg


def update_config_partial_dataset(cfg: Dict, frac: float):
    """Update config parameters to take a fraction of dataset

    Parameters
    ----------
    cfg : Dict
        config file
    frac : float
        fraction of dataset to use
    """

    pass


# def fill_nones_with_empty_dict(cfg):
    # for key in cfg:
    #     if cfg[key] is None:
    #         cfg[key] = dict()
    #     elif isinstance(cfg[key], dict) and len(cfg[key]) > 0:
    #         return fill_nones_with_empty_dict(cfg[key])
    # return cfg


def validate_cfg(cfg):
    ### input validation
    if 'flags' not in cfg or ('flags' in cfg and cfg['flags'] is None):
        cfg['flags'] = dict()

    #cfg = fill_nones_with_empty_dict(cfg)

    return cfg


def run_optim(cfg):
    ### Single optimization run

    # for now, try training a single model
    # print(cfg)
    dataset = PytorchDataset(**cfg['training_cfg']['data_cfg'])
    model = Trainer(**cfg['training_cfg'])
    learner = pl.Trainer(**cfg['flags'])
    learner.fit(model, dataset)


def main():
    cfg = load_cfg(config_path + config_name)
    cfg = validate_cfg(cfg)
    run_optim(cfg)


if __name__ == '__main__':
    main()
