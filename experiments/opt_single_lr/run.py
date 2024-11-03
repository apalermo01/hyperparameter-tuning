from hparam_tuning_project.optimization_modules.learning_rate_pl import tune_lr
import os
import yaml
import json
from ray import tune


def main(cfg):

    search_space = {
        'optimizer_cfg': {
            'args': {
                'lr': tune.loguniform(1e-6, 1e-1)
            }
        }
    }

    best_lrs = tune_lr(cfg=cfg, num_samples=10, search_space=search_space)

    return {
        'model_id': cfg['model_cfg']['model_id'],
        'split_perc': split,
        'best_lrs': best_lrs
    }


if __name__ == '__main__':
    results_dir = "experiments/opt_single_lr/results.json"
    if os.path.exists(results_dir):
        with open(results_dir) as f:
            results = json.load(f)
    else:
        results = {}

    for path in os.listdir("experiments/opt_single_lr/training_configs/"):
        if path[0] == '_':
            continue

        with open(os.path.join("./experiments/opt_single_lr/training_configs/", path), "r") as f:
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

        with open('experiments/opt_single_lr/results.json') as f:
            json.dump(results, f, indent=2)
