
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import air, tune
from hparam_tuning_project.training.utils import build_and_fit_modules
import os
from typing import Dict


def train_func(cfg: Dict,):
    """Single optimization run"""
    tune_callback = TuneReportCallback({'val_loss': 'val_loss'},
                                       on='validation_end')

    build_and_fit_modules(cfg, extra_callbacks=[tune_callback])


def run_optim_resume(path: str):
    """Resumes training from a given path"""
    tuner = tune.Tuner.restore(os.path.abspath(path))
    results = tuner.fit()
    return results


def _run_optim_lr(cfg: Dict,
                  run_id: str,
                  max_epochs: int = 10,
                  num_samples: int = 100,
                  output_dir: str = './hparam_results/',
                  n_cpus: int = 4,
                  n_gpus: int = 0,
                  min_lr: float = 1e-6,
                  max_lr: float = 1e-1):
    """Runs lr optimization

    Parameters
    ----------
    cfg : Dict
        config describing model building and training pipeline
    run_id : str
        experiment identifier
    max_epochs : int, optional
        maximum number of epochs to run, by default 10
    num_samples : int, optional
        number of trials, by default 100
    output_dir : str, optional
        where to store results, by default './hparam_results/'
    n_cpus : int, optional
        number of cpus to use, by default 4
    n_gpus : int, optional
        number of gpus to use, by default 0
    min_lr : float, optional
        minimum lr to uise, by default 1e-6
    max_lr : float, optional
        maximum lr to use, by default 1e-1

    Returns
    -------
    ray tune results
    """
    ### init search space
    cfg['optimizer_cfg']['args']['lr'] = tune.loguniform(min_lr, max_lr)

    ### flags
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
            run_id: str,
            max_epochs: int = 10,
            num_samples: int = 100,
            output_dir: str = '../hparam_results/',
            n_cpus: int = 4,
            n_gpus: int = 0,
            min_lr: float = 1e-6,
            max_lr: float = 1e-1):
    """Run full hyperparameter tuning experiment, saves results to a csv

    Parameters
    ----------
    cfg : Dict
        config describing model building and training pipeline
    run_id : str
        experiment identifier
    max_epochs : int, optional
        maximum number of epochs to run, by default 10
    num_samples : int, optional
        number of trials, by default 100
    output_dir : str, optional
        where to store results, by default './hparam_results/'
    n_cpus : int, optional
        number of cpus to use, by default 4
    n_gpus : int, optional
        number of gpus to use, by default 0
    min_lr : float, optional
        minimum lr to uise, by default 1e-6
    max_lr : float, optional
        maximum lr to use, by default 1e-1
    """
    if os.path.exists(os.path.join(output_dir, run_id)):
        cfg['meta']['resumed'] = True
        results = run_optim_resume(os.path.join(output_dir, run_id))

    else:
        cfg['meta']['resumed'] = False
        results = _run_optim_lr(cfg, run_id, max_epochs, num_samples,
                                output_dir, n_cpus, n_gpus, min_lr, max_lr)

    df = results.get_dataframe()
    path = os.path.join(output_dir, run_id, f"results_{run_id}.csv")
    df.to_csv(path)
    print(f"results saved to {path}")
