#!/bin/bash

# run learning rate optimization for mnist and cifar datasets
# using different dataset subsets

#splits=(mnist mnist_0_75 mnist_0_5 mnist_0_25 mnist_0_1)
#configs=(cnn2_cifar10.yaml pytorch_classifier_cifar10.yaml pytorch_classifier_mnist.yaml)
#model_ids=(cnn2 resnet18 resnet18)

models=(cnn2 pytorch_classifier pytorch_classifier)
model_ids=(cnn2 resnet18 resnet18)
split_ids=(_ _0_75 _0_5 _0_25 _0_1)
datasets=(cifar10 cifar10 mnist)

for s_id in ${split_ids[@]}
do
    # echo ${split_ids[$split_idx]}
    for config_idx in ${!models[@]}
    do
        config_name=${models[$config_idx]}_${datasets[$config_idx]}.yaml
        
        if [[ "$s_id" == "_" ]]; then
            split_id=${datasets[$config_idx]}
        else
            split_id=${datasets[$config_idx]}${s_id}
        fi
        
        run_id=lr_opt_20221217_${model_ids[$config_idx]}_${split_id}

        echo "config name: $config_name, split id: $split_id, run_id: $run_id"
        python scripts/experiment_runner_lr_1.py \
            --config_name $config_name \
            --run_id $run_id \
            --split_id $split_id \
            --test_mode
    done
done
# for split_id in "${splits[@]}"
# do
#     for i in "${!configs[@]}"
#     do
#         echo "running split $split_id, config name is ${configs[$i]}, model id is ${model_ids[$i]}"
#         python scripts/experiment_runner_lr_1.py \
#              --config_name ${configs[$i]} \
#              --run_id lr_opt_20221217_${model_ids[$i]}_${split_id} \
#              --split_id ${split_id} \
#              --test_mode
# 
#     done
# done