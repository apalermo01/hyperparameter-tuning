"""Functionality to run one iteration of the experiment"""
import argparse
from typing import Dict
import yaml
import os
from hparam_tuning_project.utils import load_cfg
from hparam_tuning_project.optimization_funcs import tune_lr

config_root = "/home/alex/Documents/personal-projects/hyperparameter-tuning/training_configs/"


def main():
    args = parse_args()
    cfg = load_cfg(args.config_root + args.config_name, args)

    split_id = args.split_id
    if split_id is not None:
        cfg['data_cfg']['split_id'] = split_id
        print("[INFO] setting split id in config to ", split_id)

    tune_lr(cfg,
            run_id=args.run_id,
            max_epochs=args.max_epochs,
            num_samples=args.num_samples,
            local_dir=args.local_dir,
            n_cpus=args.n_cpus,
            n_gpus=args.n_gpus,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            test=args.test)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_root', default=config_root)
    parser.add_argument('--config_name')
    parser.add_argument('--run_id', required=True)

    parser.add_argument("--max_epochs", default=10)
    parser.add_argument("--num_samples", default=100)
    parser.add_argument("--local_dir", default="../hparam_results/")
    parser.add_argument("--n_cpus", default=2)
    parser.add_argument('--n_gpus', default=0)
    parser.add_argument("--min_lr", default=1e-6)
    parser.add_argument("--max_lr", default=1e-1)
    parser.add_argument("--split_id", default=None)
    parser.add_argument("--test", action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
