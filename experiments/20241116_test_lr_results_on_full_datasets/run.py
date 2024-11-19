from hparam_tuning_project.training.utils import build_modules
import yaml
import os
from tqdm import tqdm
from copy import deepcopy


def main():
    root_path = "./experiments/20241109_lr_batch/results/"
    results_path = "./experiments/20241116_test_lr_results_on_full_datasets/results/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for file in tqdm(os.listdir(root_path)):
        if ".yaml" not in file and ".yml" not in file:
            print(f"{file} is not a yaml file")
            continue
        new_file_name = file.split('.')[0] + '_with_train_data.yaml'
        if os.path.exists(os.path.join(results_path, new_file_name)):
            print(f"{new_file_name} already exists. Continuing...")
            continue

        print("testing " + new_file_name)
        with open(os.path.join(root_path, file), "r") as f:
            res = yaml.safe_load(f)

        cfg_prev = res['config']
        cfg_test = deepcopy(cfg_prev)

        best_lr = res['best_metrics']['config']['train_loop_config'][
            'optimizer_cfg']['args']['lr']

        cfg_test['optimizer_cfg']['args']['lr'] = best_lr
        cfg_test['data_cfg']['split_id'] = 'mnist'
        cfg_test['data_cfg']['workdir'] = \
            '/home/alex/Documents/git/hyperparameter-tuning/'
        cfg_test['data_cfg']['num_workers'] = 4
        cfg_test['data_cfg']['batch_size'] = 256
        cfg_test['flags']['default_root_dir'] = \
            './experiments/20241116_test_lr_results_on_full_datasets/'
        cfg_test['flags']['max_epochs'] = 30
        cfg_test['flags']['enable_progress_bar'] = True

        learner, model, dataset = build_modules(cfg_test)

        learner.fit(model, dataset)
        test_results = learner.test(ckpt_path='best', datamodule=dataset)
        accuracy = test_results[0]['accuracy_epoch']

        res['testing_data'] = {
            'test_accuracy': accuracy,
            'test_config': cfg_test
        }
        with open(os.path.join(results_path, new_file_name), "w") as f:
            yaml.dump(res, f)


if __name__ == '__main__':
    main()
