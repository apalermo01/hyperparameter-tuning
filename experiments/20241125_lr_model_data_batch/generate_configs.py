import yaml


def main():
    default_root_dir = "./experiments/20241125_lr_model_data_batch/training_configs/"

    datasets = ['mnist', 'cifar10', 'cifar100']
    models = [
        {
            'model_name': 'CNN-architecture-1',
            'model_cfg': {
                'model_id': 'CNN2',
                'args': {
                    'batch_norm': True,
                    'dropout': True,
                    'ch1': 8,
                    'ch2': 16,
                    'ch3': 64,
                }
            }
        },

        {
            'model_name': 'CNN-architecture-2',
            'model_cfg': {
                'model_id': 'CNN2',
                'args': {
                    'batch_norm': True,
                    'dropout': True,
                    'ch1': 32,
                    'ch2': 128,
                    'ch3': 512,
                }
            }
        },
        {
            'model_name': 'resnet18',
            'model_cfg': {
                'model_id': 'pytorch_model',
                'args': {
                    'architecture_id': 'resnet_18',
                    'pretrained': True
                }

            }
        }
    ]

    split_mappings = {
        100: '',
        75: '_0_75',
        50: '_0_5',
        25: '_0_25',
        10: '_0_1',
    }

    base_cfg = default_root_dir + '_base_cfg.yaml'

    with open(base_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    for d in datasets:
        cfg['data_cfg']['dataset_id'] = d
        print(f"writing configs for dataset {d}")
        for m in models:
            print(f"writing configs for model {m['model_name']}")
            cfg['model_cfg'] = m['model_cfg']
            if d == 'mnist':
                cfg['model_cfg']['args']['input_shape'] = [28, 28]
                cfg['model_cfg']['args']['in_channels'] = 1
                cfg['model_cfg']['args']['num_classes'] = 10

            if d == 'cifar10':
                cfg['model_cfg']['args']['input_shape'] = [32, 32]
                cfg['model_cfg']['args']['in_channels'] = 3
                cfg['model_cfg']['args']['num_classes'] = 10

            if d == 'cifar100':
                cfg['model_cfg']['args']['input_shape'] = [32, 32]
                cfg['model_cfg']['args']['in_channels'] = 3
                cfg['model_cfg']['args']['num_classes'] = 100

            for s in split_mappings.keys():
                print("writing configs for split {s}")
                cfg['data_cfg']['split_id'] = d + split_mappings[s]
                cfg['flags']['default_root_dir'] = default_root_dir
                path = f"{default_root_dir}/{m['model_name']}_{d}_{s}.yaml"
                print(f"writing config to path {path}")
                with open(path, "w") as f:
                    yaml.dump(cfg, f)


if __name__ == '__main__':
    main()
