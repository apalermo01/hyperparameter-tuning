from hparam_tuning_project.training.utils import build_modules
import yaml
import os
from tqdm import tqdm


def main():
    root_path = "./experiments/lr_batch_20241109/results/"
    results_path = "./experiments/test_lr_results_on_full_datasets/results/"

    for file in tqdm(os.listdir(root_path)):
        new_file_name = file.split('.')[0] + '_with_train_data.yaml'
        if os.path.exists(os.path.join(results_path, new_file_name)):
            print(f"{new_file_name} already exists. Continuing...")
            continue

        print("testing " + new_file_name)
        with open(os.path.join(root_path, file), "r") as f:
            res = yaml.safe_load(f)

        cfg = res['config']

        best_lr = res['best_metrics']['config']['train_loop_config'][
            'optimizer_cfg']['args']['lr']

        cfg['optimizer_cfg']['args']['lr'] = best_lr
        cfg['data_cfg']['split_id'] = 'mnist'
        cfg['data_cfg']['workdir'] = \
            '/home/alex/Documents/git/hyperparameter-tuning/'
        cfg['flags']['default_root_dir'] = \
            './experiments/test_lr_results_on_full_datasets/'
        cfg['flags']['max_epochs'] = 20
        cfg['flags']['enable_progress_bar'] = True

        learner, model, dataset = build_modules(cfg)

        learner.fit(model, dataset)
        test_results = learner.test(ckpt_path='best', datamodule=dataset)
        accuracy = test_results[0]['accuracy_epoch']
        res['testing_data'] = {
            'test_accuracy': accuracy,
            'test_config': cfg
        }
        with open(os.path.join(results_path, new_file_name), "w") as f:
            yaml.dump(res, f)



if __name__ == '__main__':
    main()
