"""
Run hyperparameter tuning with  pytorch-lightning with several combinations of models and datasets
"""
from hparam_tuning_project.utils import load_cfg
from hparam_tuning_project.optimization_modules.learning_rate_pl import tune_lr
import os
import json
split_mappings = {
    100: '',
    75: '_0_75',
    50: '_0_5',
    25: '_0_25',
    10: '_0_1',
}
cfg_path = "./training_configs/"


def run_opt(model_id: str,
            dataset_id: str,
            num_samples: int,
            results: list) -> dict:

    splits = [100, 75, 50, 25, 10]
    cfg = load_cfg(path=os.path.join(
        cfg_path, f"{model_id}_{dataset_id}.yaml"))
    for s in splits:
        split_id = dataset_id + split_mappings[s]
        cfg['data_cfg']['split_id'] = split_id
        best_lrs = tune_lr(cfg=cfg, num_samples=num_samples)
        results.append({
            'dataset_id': dataset_id,
            'model_id': model_id,
            'split_id': split_id,
            'split_perc': s,
            'best_lrs': best_lrs,
        })

    return results


def main():
    models = ['cnn2', 'pytorch_classifier']
    datasets = ['mnist', 'cifar10']
    num_samples = 10
    results = []
    for m in models:
        for d in datasets:
            results = run_opt(model_id=m,
                              dataset_id=d,
                              num_samples=num_samples,
                              results=results)

            with open(f"./run_results/pl_lr_optim-model={m}-dataset={d}_20241102.json", "w") as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
