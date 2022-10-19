"""Functionality to run one iteration of the experiment"""

import argparse
from distutils.command.build import build
from typing import Dict
import yaml
from hparam_tuning_project.training.trainer import Trainer
from hparam_tuning_project.data.datasets import PytorchDataset
from hparam_tuning_project.training import build_and_fit_modules
import pytorch_lightning as pl
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import os
from datetime import datetime as dt


config_root = "/home/alex/Documents/personal-projects/hyperparameter-tuning/training_configs/"
# config_name = "simple_training.yaml"
# config_name = 'pytorch_classifier_mnist.yaml'


def load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def validate_cfg(cfg):
    ### input validation
    if 'flags' not in cfg or ('flags' in cfg and cfg['flags'] is None):
        cfg['flags'] = dict()

    return cfg


def train_func(cfg, checkpoint_dir=None):
    """Single optimization run"""
    tune_callback = TuneReportCallback({'val_loss': 'val_loss'}, on='validation_end')

    build_and_fit_modules(cfg, extra_callbacks=[tune_callback])


def run_optim(cfg, run_id,):

    ### init search space
    cfg['optimizer_cfg']['args']['lr'] = tune.loguniform(1e-6, 1e-1)

    ### flags
    cfg['flags']['enable_progress_bar'] = False
    cfg['flags']['max_epochs'] = 10

    ### ray tune
    num_epochs = 10

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
        num_samples=100,
    )

    run_config = air.RunConfig(
        name=run_id,
        progress_reporter=reporter,
        local_dir='../hparam_results/'
    )

    train_fn = tune.with_resources(train_func, {'cpu': 4})

    tuner = tune.Tuner(train_fn,
                       tune_config=tune_cfg,
                       param_space=cfg,
                       run_config=run_config)
    tuner.fit()


def run_optim_resume(run_id):
    print('CONTINUING RUN', run_id)
    tuner = tune.Tuner.restore(os.path.abspath("../hparam_results/cnn2_mnist_lr_optim_20221019"))
    tuner.fit()


def main():
    args = parse_args()
    cfg = load_cfg(args.config_root + args.config_name)
    cfg = validate_cfg(cfg)

    cfg['meta'] = {
        'entry_script': 'experiment_runner_lr_1.py',
        'start_time': str(dt.now()),
        'resumed': False
    }

    if os.path.exists(f'../hparam_results/{args.run_id}'):
        cfg['meta']['resumed'] = True
        run_optim_resume(args.run_id)
    else:
        run_optim(cfg, args.run_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_root', default=config_root)
    parser.add_argument('--config_name')
    parser.add_argument('--run_id', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
