from hparam_tuning_project.training.utils import build_modules
from typing import Dict
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
    RayDDPStrategy
)
from ray.train import RunConfig, CheckpointConfig


def train_func(config):
    cfg = config['cfg']
    extra_callbacks = config['extra_callbacks']
    extra_callbacks.append(RayTrainReportCallback())
    plugins = [RayLightningEnvironment()]
    learner, model, dataset = build_modules(
        cfg=cfg,
        extra_callbacks=extra_callbacks,
        plugins=plugins,
        strategy=RayDDPStrategy()
    )
    learner = prepare_trainer(learner)
    print("learner = ", learner)
    print("model = ", model)
    learner.fit(model, dataset)


def tune_lr(cfg: Dict,
            search_space: Dict,
            num_samples: int = 10,
            num_epochs=10):

    cfg['flags']['enable_progress_bar'] = False

    extra_callbacks = []
    # best_lrs = []
    train_loop_config = {
        'cfg': cfg,
        'extra_callbacks': extra_callbacks
    }

    # for _ in range(num_samples):
    scheduler = ASHAScheduler(max_t=num_epochs,
                              grace_period=1,
                              reduction_factor=2)
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max"
        )
    )
    ray_trainer = TorchTrainer(
        train_func,
        train_loop_config=train_loop_config,
        run_config=run_config
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler
        )
    )

    tuner.fit()
