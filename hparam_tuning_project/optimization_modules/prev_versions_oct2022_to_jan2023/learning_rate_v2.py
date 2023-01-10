"""
Second iteration of learning rate optimization
includes functionality for multiple sampling methods
"""

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import air, tune
from hparam_tuning_project.training.utils import build_and_fit_modules
import os
from typing import Dict
import numpy as np


def train_func(cfg: Dict, test_mode=False):
    """Single optimization run"""
    if test_mode:
        return build_and_fit_modules(cfg)

    tune_callback = TuneReportCallback({'val_loss': 'val_loss'},
                                       on='validation_end')

    build_and_fit_modules(cfg, extra_callbacks=[tune_callback])


def run_optim_resume(path: str):
    """Resumes training from a given path"""
    tuner = tune.Tuner.restore(os.path.abspath(path))
    results = tuner.fit()
    return results


def _init_search_space(cfg: Dict,
                       min_lr: float = 1e-6,
                       max_lr: float = 1e-1,
                       num_grid_search_samples: int = 10,
                       method: str = 'log_random') -> Dict:
    assert method in ['log_random', 'grid_search']
    if method == 'log_random':
        cfg['optimizer_cfg']['args']['lr'] = tune.loguniform(min_lr, max_lr)
    elif method == 'grid_search':
        cfg['optimizer_cfg']['args']['lr'] = tune.grid_search(np.logspace(min_lr, max_lr, num_grid_search_samples))
    return cfg


def _run_optim_lr(cfg: Dict,
                  lr_args: Dict,
                  run_id: str,
                  max_epochs: int = 10,
                  num_samples: int = 100,
                  output_dir: str = "./hparam_results/",
                  n_cpus: int = 4,
                  n_gpus: int = 0,
                  test_mode: bool = False,):
    if test_mode:
        cfg['flags']['fast_dev_run'] = True
        train_func(cfg, test_mode=True)
        return

    cfg = _init_search_space(cfg, **lr_args)

    cfg['flags']['enable_progress_bar'] = False
    cfg['flags']['max_epochs'] = max_epochs

    scheduler = ASHAScheduler(
        max_t=max_epochs,
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
        num_samples=num_samples,
    )

    run_config = air.RunConfig(
        name=run_id,
        progress_reporter=reporter,
        local_dir=output_dir,
    )

    train_fn = tune.with_resources(train_func, {'cpu': n_cpus, 'gpu': n_gpus})
    tuner = tune.Tuner(train_fn,
                       tune_config=tune_cfg,
                       param_space=cfg,
                       run_config=run_config)
    results = tuner.fit()
    return results


def tune_lr(cfg: Dict,
            lr_args: Dict,
            run_args: Dict,
            test_mode: bool = False,
            force_restart: bool = False,
            ):

    if os.path.exists(os.path.join(run_args['output_dir'], run_args['run_id'])) and not force_restart and not test_mode:
        cfg['meta']['resumed'] = True
        results = run_optim_resume(os.path.join(run_args['output_dir'], run_args['run_id']))
    else:
        cfg['meta']['resumed'] = False
        results = _run_optim_lr(cfg=cfg, lr_args=lr_args, test_mode=test_mode, **run_args)
    if test_mode:
        return
    df = results.get_dataframe()
    path = os.path.join(run_args['output_dir'], run_args['run_id'], f"results_{run_args['run_id']}.csv")
    df.to_csv(path)
    print(f"results saved to {path}")
