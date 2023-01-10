"""
Second version of learning rate experiment script -> moves the arguments for learning rate
into a dictionary, along with some other minor refactoring
"""


import argparse
from hparam_tuning_project.utils import load_cfg
from hparam_tuning_project.optimization_modules.learning_rate_v2 import tune_lr


config_root = "./training_configs/"


def main():
    args = parse_args()
    cfg = load_cfg(args.config_root + args.config_name, args)
    if len(args.split_id) > 0:
        cfg['data_cfg']['split_id'] = args.split_id

    lr_args = dict(
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_grid_search_samples=args.num_grid_search_samples,
        method=args.search_method,
    )

    run_args = dict(
        run_id=args.run_id,
        max_epochs=args.max_epochs,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        n_cpus=args.n_cpus,
        n_gpus=args.n_gpus,
    )

    tune_lr(cfg=cfg,
            lr_args=lr_args,
            run_args=run_args,
            force_restart=args.force_restart,
            test_mode=args.test_mode)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_root', default=config_root)
    parser.add_argument('--config_name', default='cnn2_mnist.yaml')
    parser.add_argument('--run_id', default='test_run')

    parser.add_argument("--max_epochs", default=10)
    parser.add_argument("--num_samples", default=100)
    parser.add_argument("--output_dir", default="../hparam_results/")
    parser.add_argument("--n_cpus", default=2)
    parser.add_argument('--n_gpus', default=0)
    parser.add_argument("--min_lr", default=1e-6)
    parser.add_argument("--max_lr", default=1e-1)
    parser.add_argument("--num_grid_search_samples", default=20)
    parser.add_argument("--search_method", default='log_random')
    parser.add_argument('--split_id', default='')
    parser.add_argument("--force_restart", action="store_true")
    parser.add_argument('--test_mode', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
