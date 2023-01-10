import argparse
from hparam_tuning_project.utils import load_cfg
from hparam_tuning_project.optimization_modules.learning_rate_v3 import tune_lr

config_root = "./training_configs/"

def main():
    args = parse_args()

    for split in args.split_ids:
        print(split)
    return 
    cfg = load_cfg(args.config_root + args.config_name, args)
        cfg['data_cfg']['split_id'] = args.split_id

    run_args = dict(
            run_id=args.run_id,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
    )

    lr_args = {}
    tune_lr(cfg=cfg,
            lr_args=lr_args,
            run_args=run_args,
            force_restart=args.force_restart,
            test_mode=args.test_mode)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_root', default=config_root)
    parser.add_argument('--config_name', default='cnn2_mnist.yaml')
    parser.add_argument('--run_id', default='test_run')
    parser.add_argument("--num_samples", default=100)
    parser.add_argument("--output_dir", default="../hparam_results/")
    parser.add_argument('--splits', required=True, nargs='+')
    parser.add_argument("--force_restart", action="store_true")
    parser.add_argument('--test_mode', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
