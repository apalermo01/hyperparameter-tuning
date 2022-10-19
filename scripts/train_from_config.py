import argparse
from distutils.command.build import build
import yaml
from hparam_tuning_project.training import build_and_fit_modules
import os
from datetime import datetime as dt


def main():
    args = parse_args()
    if '.yaml' not in args.config_name:
        config_name = f"{args.config_name}.yaml"
    else:
        config_name = args.config_name

    with open(os.path.join(args.config_path, config_name), 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['meta'] = {
        'entry_script': 'train_from_config.py',
        'start_time': str(dt.now())
    }

    build_and_fit_modules(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='')
    parser.add_argument('--config_name', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
