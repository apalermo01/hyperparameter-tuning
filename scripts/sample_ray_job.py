"""
Attempt at submitting a model training job to a local ray cluster
"""
from hparam_tuning_project.utils import load_cfg
from hparam_tuning_project.optimization_modules.learning_rate_v2 import tune_lr
import os
config_root = "./training_configs"


def main():
    cfg = load_cfg(os.path.join(config_root, "sample_job.yaml"))

    lr_args = dict(
        min_lr=1e-4,
        max_lr=1e-1,
        num_grid_search_samples=2,
        method="grid_search",
    )

    run_args = dict(
        run_id="sample_job",
        max_epochs=3,
        num_samples=10,
        output_dir="../hparam_sample_results",
        n_cpus=1,
        n_gpus=0,
    )

    tune_lr(cfg=cfg,
            lr_args=lr_args,
            run_args=run_args,
            force_restart=False,
            test_mode=False,)

if __name__ == '__main__':
    main()
