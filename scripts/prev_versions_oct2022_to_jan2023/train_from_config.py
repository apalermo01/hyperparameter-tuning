import argparse
import yaml
from hparam_tuning_project.training.utils import build_and_fit_modules
from hparam_tuning_project.utils import load_cfg
import os
from datetime import datetime as dt


def main():
    args = parse_args()
    if '.yaml' not in args.config_name:
        config_name = f"{args.config_name}.yaml"
    else:
        config_name = args.config_name

    cfg = load_cfg(os.path.join(args.config_path, config_name), args)

    build_and_fit_modules(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./training_configs/')
    parser.add_argument('--config_name', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
