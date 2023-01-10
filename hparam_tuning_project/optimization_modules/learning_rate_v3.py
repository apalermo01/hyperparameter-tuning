"""
Third iteration of learning rate optimization
Uses the pytorch lightning learning rate finder to find the best learning rate
"""

from hparam_tuning_project.training.utils import build_modules
import os
from typing import Dict
import numpy as np


def _run_optim_lr(cfg: Dict,
                  lr_args: Dict,
                  run_id: str,
                  output_dir: str = "./hparam_results/",
                  num_samples: int = 10,
                  n_cpus: int = 1,
                  n_gpus: int = 0,
                  test_mode: bool = False,):

    cfg['flags']['enable_progress_bar'] = False
    extra_callbacks = []
    best_lrs = []

    for _ in range(num_samples):
        learner, model, dataset = build_modules(cfg=cfg, extra_callbacks=extra_callbacks)
        lr_finder = learner.tuner.lr_find(model, dataset)
        best_lrs.append(lr_finder.suggestion())
    return best_lrs

def tune_lr(cfg: Dict,
            lr_args: Dict,
            run_args: Dict,
            test_mode: bool = False,
            force_restart: bool = False,):

    cfg['meta']['resumed'] = False
    if os.path.exists(os.path.join(run_args['output_dir'], run_args['run_id'])) and not force_restart and not test_mode:
        raise NotImplementedError
    else:
        results = _run_optim_lr(cfg=cfg,
                                lr_args=lr_args,
                                test_mode=test_mode,
                                **run_args,)

        return results
