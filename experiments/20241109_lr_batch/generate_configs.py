import yaml


def main():

    split_mappings = {
        100: '',
        75: '_0_75',
        50: '_0_5',
        25: '_0_25',
        10: '_0_1',
    }

    cfg_path = 'experiments/20241109_lr_batch/training_configs/'
    base_cfg = cfg_path + '_pytorch_classifier_mnist.yaml'
    with open(base_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    for s in split_mappings.keys():
        dataset_id = cfg['data_cfg']['dataset_id']
        cfg['data_cfg']['split_id'] = dataset_id + split_mappings[s]
        with open(f"{cfg_path}/pytorch_classifier_mnist_{s}.yaml", "w") as f:
            yaml.dump(cfg, f)


if __name__ == '__main__':
    main()
