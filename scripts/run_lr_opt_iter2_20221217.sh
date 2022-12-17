#!/bin/bash

# run learning rate optimization for mnist and cifar datasets
# using different dataset subsets

splits=(mnist mnist_0_75 mnist_0_5 mnist_0_25 mnist_0_1)
configs=(cnn2_cifar10.yaml, pytorch_classifier_cifar10.yaml)

for split_id in "${splits[@]}"
do
    for config_id in "${configs[@]}"
    do
        echo "running split id $split_id with config $config_id"

    done
done