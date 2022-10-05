"""Functionality to run one iteration of the experiment"""

from typing import Dict
import yaml
from hparam_tuning_project.training.trainer import Trainer
from hparam_tuning_project.data.datasets import PytorchDataset
from hparam_tuning_project.utils import initialize_callbacks
import pytorch_lightning as pl
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter


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


def validate_cfg(cfg):
    ### input validation
    if 'flags' not in cfg or ('flags' in cfg and cfg['flags'] is None):
        cfg['flags'] = dict()

    # cfg = fill_nones_with_empty_dict(cfg)

    return cfg


def train_func(cfg):
    ### Single optimization run
    tune_callback = TuneReportCallback('val_loss', on='validation_end')

    # for now, try training a single model
    dataset = PytorchDataset(**cfg['training_cfg']['data_cfg'])
    model = Trainer(**cfg['training_cfg'])
    callbacks = initialize_callbacks(cfg['callbacks'])
    callbacks.append(tune_callback)
    print(callbacks)
    learner = pl.Trainer(**cfg['flags'], callbacks=callbacks)
    learner.fit(model, dataset)


def run_optim(cfg):
    cfg['training_cfg']['optimizer_cfg']['args']['lr'] = tune.loguniform(1e-4, 1e-1)
    cfg['flags']['enable_progress_bar'] = False
    num_epochs = 2
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=['val_loss', 'training_iteration']
    )

    tune_cfg = tune.TuneConfig(
        metric='val_loss',
        mode='min',
        scheduler=scheduler,
        num_samples=2,
    )

    # run_config = air.RunConfig(
    #     name='test',
    #     progress_reporter=reporter
    # )
    train_fn = tune.with_parameters(train_func)

    # tuner = tune.Tuner(train_fn,
    #                    tune_config=tune_cfg,
    #                    param_space=cfg,
    #                    run_config=run_config)
    results = tune.run(train_fn,
                       resources_per_trial={"cpu": 8, "gpu": 1},
                       metric='val_loss',
                       mode='min',
                       config=cfg,
                       num_samples=2,
                       scheduler=scheduler,
                       progress_reporter=reporter,
                       name='test')
    # results = tuner.fit()


def main():
    cfg = load_cfg(config_path + config_name)
    cfg = validate_cfg(cfg)

    run_optim(cfg)


if __name__ == '__main__':
    main()
