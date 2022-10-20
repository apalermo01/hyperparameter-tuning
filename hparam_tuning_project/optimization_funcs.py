from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import air, tune
from hparam_tuning_project.training.utils import build_and_fit_modules
import os


def train_func(cfg, checkpoint_dir=None):
    """Single optimization run"""
    tune_callback = TuneReportCallback({'val_loss': 'val_loss'}, on='validation_end')

    build_and_fit_modules(cfg, extra_callbacks=[tune_callback])


def run_optim_resume(path):
    tuner = tune.Tuner.restore(os.path.abspath(path))
    results = tuner.fit()
    return results


def _run_optim_lr(cfg,
                  run_id,
                  max_epochs=10,
                  num_samples=100,
                  local_dir='./hparam_results/',
                  n_cpus=4,
                  n_gpus=0,
                  min_lr=1e-6,
                  max_lr=1e-1):

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
        local_dir=local_dir
    )

    train_fn = tune.with_resources(train_func, {'cpu': n_cpus, 'gpu': n_gpus})

    tuner = tune.Tuner(train_fn,
                       tune_config=tune_cfg,
                       param_space=cfg,
                       run_config=run_config)
    results = tuner.fit()
    return results


def _run_optim_cnn2_structure(cfg,
                              run_id,
                              max_epochs=10,
                              num_samples=100,
                              local_dir='./hparam_results/',
                              n_cpus=4,
                              n_gpus=0,
                              in_channels=1,
                              n_classes=10):

    ### init search space
    # cfg['optimizer_cfg']['args']['lr'] = tune.loguniform(min_lr, max_lr)
    # cfg['optimizer_cfg']['']
    assert cfg['model_cfg']['model_id'] == 'CNN2'
    cfg['model_cfg']['args'] = {
        'in_channels': in_channels,
        'n_classes': n_classes,
        'ch1': tune.choice([4, 8]),
        'ch2': tune.choice([16, 32, 64]),
        'ch3': tune.choice([64, 128, 512]),
        'lin1': tune.choice([4096, 1024, 512, 128]),
        'batch_norm': tune.choice([True, False]),
    }

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
        local_dir=local_dir
    )

    train_fn = tune.with_resources(train_func, {'cpu': n_cpus, 'gpu': n_gpus})

    tuner = tune.Tuner(train_fn,
                       tune_config=tune_cfg,
                       param_space=cfg,
                       run_config=run_config)
    results = tuner.fit()
    return results


def tune_lr(cfg,
            run_id,
            max_epochs=10,
            num_samples=100,
            local_dir='../hparam_results/',
            n_cpus=4,
            n_gpus=0,
            min_lr=1e-6,
            max_lr=1e-1):

    if os.path.exists(os.path.join(local_dir, run_id)):
        cfg['meta']['resumed'] = True
        results = run_optim_resume(os.path.join(local_dir, run_id))

    else:
        cfg['meta']['resumed'] = False
        results = _run_optim_lr(cfg, run_id, max_epochs, num_samples,
                                local_dir, n_cpus, n_gpus, min_lr, max_lr)

    df = results.get_dataframe()
    path = os.path.join(local_dir, run_id, f"results_{run_id}.csv")
    df.to_csv(path)
    print(f"results saved to {path}")


# def tune_optim_cnn2_structure(cfg,
#                               run_id,
#                               max_epochs=10,
#                               num_samples=100,
#                               local_dir='../hparam_results/',
#                               n_cpus=4,
#                               n_gpus=0,
#                               min_lr=1e-6,
#                               max_lr=1e-1):

#     if os.path.exists(os.path.join(local_dir, run_id)):
#         cfg['meta']['resumed'] = True
#         results = run_optim_resume(os.path.join(local_dir, run_id))

#     else:
#         cfg['meta']['resumed'] = False
#         results = _run_optim_adam(cfg, run_id, max_epochs, num_samples,
#                                 local_dir, n_cpus, n_gpus, min_lr, max_lr)

#     df = results.get_dataframe()
#     path = os.path.join(local_dir, run_id, f"results_{run_id}.csv")
#     df.to_csv(path)
#     print(f"results saved to {path}")
