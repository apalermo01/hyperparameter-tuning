from hparam_tuning_project.training.utils import build_modules
from typing import Dict
import ray
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
import time


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
    learner.fit(model, dataset)


def run_tuner(cfg: Dict,
              search_space: Dict,
              num_samples: int = 10,
              num_epochs: int = 10,
              ret_tuner: bool = False):

    ray.init(_memory=12 * 1024 ** 3)

    cfg['flags']['enable_progress_bar'] = False

    extra_callbacks = []

    # best_lrs = []
    train_loop_config = {
        'cfg': cfg,
        'extra_callbacks': extra_callbacks
    }

    # for _ in range(num_samples):
    scheduler = ASHAScheduler(max_t=num_epochs,
                              grace_period=3,
                              reduction_factor=2)
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
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
            metric="val_loss",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=4,
            reuse_actors=True
        )
    )

    tuner.fit()

    results = {
        'epoch_metrics': tuner.get_results().get_dataframe().to_dict(
            orient='index'),
        'best_metrics': tuner.get_results().get_best_result().metrics,

        'iteration_metrics': {r.metrics.get('experiment_tag', f'unknown_experiment_tag_{i}'):
                              r.metrics_dataframe.to_dict(orient='index')
                              for i, r in enumerate(tuner.get_results())}
    }

    ray.shutdown()
    time.sleep(15)

    if ret_tuner:
        return results, tuner
    else:
        return results
