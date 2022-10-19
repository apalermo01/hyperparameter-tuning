from distutils.command.build import build
import yaml
from hparam_tuning_project.training import build_and_fit_modules


if __name__ == '__main__':
    with open('./training_configs/pytorch_classifier_mnist.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['model_cfg']['args']['architecture_id'] = 'regnet_y_400mf'

    build_and_fit_modules(cfg)
