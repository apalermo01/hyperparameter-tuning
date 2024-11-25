import yaml


def main():
    default_root_dir = "./experiments/20241125_lr_model_data_batch/"

    cfg_bases = [
        "_pytorch_classified_mnist.yaml",
        "_pytorch_classifier_cifar10.yaml",
        "_pytorch_classifier_caltech101.yaml"
    ]

    datasets = ['mnist', 'cifar10', 'caltech101']
    split_mappings = {
        100: '',
        75: '_0_75',
        50: '_0_5',
        25: '_0_25',
        10: '_0_1',
    }

    cfg_path = 'experiments/20241125_model_data_batch/training_configs/'

    for b, d in zip(cfg_bases, datasets):
        base_cfg = cfg_path + b
        with open(base_cfg, "r") as f:
            cfg = yaml.safe_load(f)

        for s in split_mappings.keys():
            dataset_id = cfg['data_cfg']['dataset_id']
            cfg['data_cfg']['split_id'] = dataset_id + split_mappings[s]
            cfg['flags']['default_root_dir'] = default_root_dir
            with open(f"{cfg_path}/pytorch_classifier_{d}_{s}.yaml", "w") as f:
                yaml.dump(cfg, f)


if __name__ == '__main__':
    main()
