"""
Third iteration of learning rate optimization
Uses the pytorch lightning learning rate finder to find the best learning rate
"""

from hparam_tuning_project.training.utils import build_modules
import os
from typing import Dict
import numpy as np


def tune_lr(cfg: Dict,
            num_samples: int = 10,
            ):

    cfg['flags']['enable_progress_bar'] = False
    extra_callbacks = []
    best_lrs = []

    for _ in range(num_samples):
        learner, model, dataset = build_modules(
            cfg=cfg, extra_callbacks=extra_callbacks)
        lr_finder = learner.tuner.lr_find(model, dataset)
        best_lr = float(lr_finder.suggestion())
        best_lrs.append(best_lr)

    return best_lrs
