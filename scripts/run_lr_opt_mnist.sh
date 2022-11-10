#!/bin/bash

# run lr optimization for mnist dataset using different dataset sizes
splits=(mnist mnist_0_75 mnist_0_5 mnist_0_25 mnist_0_1)

for split_id in "${splits[@]}"
do
    python scripts/experiment_runner_lr_1.py \
        --config_name cnn2_mnist.yaml \
        --run_id lr_opt_20221109_${split_id}
        --root_dir linode_path
done