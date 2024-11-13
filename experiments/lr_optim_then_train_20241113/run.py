from hparam_tuning_project.optimization_modules.ray_optimize import run_tuner, train_func
import yaml
import os


def main():
    root_path = "./experiments/lr_batch_20241109/results/"
    for file in os.listdir(root_path):
        with open(os.path.join(root_path, file), "r") as f:
            res = yaml.safe_load(f)
        cfg = res['config']

        best_lr = res['best_metrics']['config']['train_loop_config'][
            'optimizer_cfg']['args']['lr']

        cfg['optimizer_cfg']['args']['lr'] = best_lr
        cfg['data_cfg']['split_id'] = 'mnist'

        print(cfg['data_cfg']['split_id'] + ", " +
              str(cfg['optimizer_cfg']['args']['lr']))

        learner = train_func(cfg, ret_learner=True)
        learner.test(chkpt_path='best')


if __name__ == '__main__':
    main()
