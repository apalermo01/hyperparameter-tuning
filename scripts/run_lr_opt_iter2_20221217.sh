#!/bin/bash

# run learning rate optimization for mnist and cifar datasets
# using different dataset subsets

splits=(mnist mnist_0_75 mnist_0_5 mnist_0_25 mnist_0_1)
configs=(cnn2_cifar10.yaml, pytorch_classifier_cifar10.yaml, pytorch_classifier_mnist)
model_ids=(cnn2, resnet18, resnet18)

for split_id in "${splits[@]}"
do
    for i in "${!configs[@]}"
    do
        echo "running split $split_id, config name is ${configs[$i]}, model id is ${model_ids[$i]}"
        python scripts/experiment_runner_lr_1 \
             --config_name ${config_id[$i]} \
             --run_id lr_opt_20221217_${model_ids[$i]}_${split_id}
             --split_id ${split_id}


    done
done