# hyperparameter-tuning
The objective of this project is to emperically determine whether or not the best configuration obtained from running hyperparameter optimization on a subset of a dataset will also be the best configuration when training on the full dataset. If this is the case, then using a subset of the full dataset can either allow for a much more rapid optimization process or allow for a much wider search space when optimizing hyperparameters.

I just finished a proof-of-concept trial to test out the training pipeline with raytune, please see the notebook '01 - analyze results from proof-of-concept runs on 20221019' for the discussion.

# Docker setup
To set up the dependancies in a docker container, do:

build the image:
```bash
docker build --tag hparam_project .
```

start the container with a bind mount attached:
```bash
docker run -d \
           --mount
           --name hparam hparam_project tail -f /dev/null
```

open a bash terminal inside the container:
```
docker exec -it hparam bash
```

while inside the container, activate the conda environment
```bash
conda activate hparam_project
```
# TODO

- [x] build a simple feed forward network
- [x] build a simple convolutional network
- [x] implement callback handling
- [x] conduct proof-of-concept experiment comparing learning rate optimization on mnist
- [x] dockerize project for portability
- [ ] update requirements and add installation instructions
- [x] rethink tran/val splits (use stratified sampling and sample 100%, 75%, 50%, 25%, and 10% of dataset)
- [ ] implement learning rate scheduling
- [ ] change learning rate hyperparameter, and other hparams in training
- [ ] implement data augmentation


## Notes
linode -> object store:
s3cmd put /hparam_results/lr_opt_20221109_mnist/* s3://hparam-project/hparam_results/lr_opt_20221109_mnist --recursive

https://www.linode.com/docs/products/storage/object-storage/guides/s3cmd


- next steps 20221217

- cifar10 + cnn2
- mnist + resnet18
- cifar10 + resnet18