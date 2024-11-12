from hparam_tuning_project.optimization_modules.ray_optimize import run_tuner
import os
import yaml
from ray import tune


def main(cfg):

    search_space = {
        'optimizer_cfg': {
            'args': {
                'lr': tune.loguniform(1e-6, 1e-1)
            }
        }
    }

    results = run_tuner(cfg=cfg,
                        num_epochs=3,
                        num_samples=3,
                        search_space=search_space)

    return {
        'model_id': cfg['model_cfg']['model_id'],
        'dataset_id': cfg['data_cfg']['split_id'],
        'results': results
    }


if __name__ == '__main__':
    base_path = "experiments/lr_batch_20241109/"
    base_config_path = "experiments/lr_batch_20241109/training_configs/"
    results = {}
    num_trials = 10
    for path in os.listdir(base_config_path):

        if path[0] == '_':
            continue

        with open(os.path.join(base_config_path, path), "r") as f:
            cfg = yaml.safe_load(f)

        split_id = cfg['data_cfg']['split_id']
        model_id = cfg['model_cfg']['model_id']

        key_base = f"{split_id}_{model_id}"
        results_path = base_path + "results/" + f"{key_base}_results.yaml"

        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = yaml.safe_load(f)

        for i in range(num_trials):

            key = f"{key_base}_trial={i}"
            print("key = ", key)
            if key in results:
                print(f"{key} already ran. Continuing...")
                continue

            ret = main(cfg)

            ret['split_id'] = split_id
            ret['model_id'] = model_id
            ret['full_cfg'] = cfg
            ret['trial'] = i
            results[key] = ret

            with open(results_path, 'w') as f:
                yaml.dump(results, f, indent=2)
