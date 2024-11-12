from hparam_tuning_project.optimization_modules.ray_optimize import run_tuner
import os
import yaml
import json
from ray import tune
import ray


def main(cfg):

    search_space = {
        'optimizer_cfg': {
            'args': {
                'lr': tune.loguniform(1e-6, 1e-1)
            }
        }
    }

    ray.init(_memory=12 * 1024 * 3)
    results = run_tuner(cfg=cfg,
                        num_samples=2,
                        num_epochs=2,
                        search_space=search_space)

    ray.shutdown()
    return {
        'model_id': cfg['model_cfg']['model_id'],
        'split_perc': split,
        'results': results
    }


if __name__ == '__main__':
    results_dir = "experiments/basic_lr_run_20241109/results.yaml"
    base_config_path = "experiments/basic_lr_run_20241109/training_configs/"
    if os.path.exists(results_dir):
        with open(results_dir) as f:
            results = yaml.safe_load(f)
    else:
        results = {}

    for path in os.listdir(base_config_path):
        if path[0] == '_':
            continue

        with open(os.path.join(base_config_path, path), "r") as f:
            cfg = yaml.safe_load(f)

        key = path.split('.')[0]

        if key in results:
            continue

        dataset = key.split('_')[2]
        split = key.split('_')[3]

        ret = main(cfg)

        ret['dataset_id'] = dataset
        ret['split_id'] = split
        ret['full_cfg'] = cfg
        results[key] = ret

        with open(results_dir, 'w') as f:
            yaml.dump(results, f, indent=2)
