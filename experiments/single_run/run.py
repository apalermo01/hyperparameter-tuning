from hparam_tuning_project.training.utils import build_and_fit_modules
import os
import yaml
from typing import List, Dict


def main():
    cfg_root: str = "experiments/single_run/training_configs/"
    cfg_path: str = os.path.join(cfg_root, "pytorch_classifier_mnist.yaml")
    with open(cfg_path, "r") as f:
        cfg: Dict = yaml.safe_load(f)


    build_and_fit_modules(cfg=cfg)


if __name__ == '__main__':
    main()
