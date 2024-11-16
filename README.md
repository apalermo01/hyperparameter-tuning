# hyperparameter-tuning
The objective of this project is to emperically determine whether or not the best configuration obtained from running hyperparameter optimization on a subset of a dataset will also be the best configuration when training on the full dataset. If this is the case, then using a subset of the full dataset can either allow for a much more rapid optimization process or allow for a much wider search space when optimizing hyperparameters.

I just finished a proof-of-concept trial to test out the training pipeline with raytune, please see the notebook '01 - analyze results from proof-of-concept runs on 20221019' for the discussion.

# Setup

To run this project, first clone the repository.

`git clone git@github.com:apalermo01/hyperparameter-tuning.git`

create a python virtual environment using the tool of your choice.

Install the requirements.

`pip install requirements_cpu.txt`

If you use a gpu: update the requirements_cpu.txt to reflect the installation options for pytorch relevant to you.

Install the project.

`pip install --editable .`

# Project structure

This project uses pytorch lightning and ray tune to train and optimize many models trained on common datasets (e.g. MNIST). The core of the project is in the `hparam_tuning_project` folder. Developemnt and analysis notebook are located in the `notebooks` folder. Jupyter notebooks that represent active development or are relevant to the writeup are in the root directory of the project. All datasets use precomputed splits that live in the `splits` folder. Experiments live in the `experiments` folder. Each experiment has a `run.py` or `run.sh` file that is used to run the whole experiment, a `results` folder that stores the results from each experiment in yaml format, and optionally a `training_configs` folder that has the configuration options for each model / dataset split.

# Results



# TODO

- [x] build a simple feed forward network
- [x] build a simple convolutional network
- [x] implement callback handling
- [x] conduct proof-of-concept experiment comparing learning rate optimization on mnist
- [x] dockerize project for portability
- [ ] update requirements and add installation instructions
- [x] rethink tran/val splits (use stratified sampling and sample 100%, 75%, 50%, 25%, and 10% of dataset)
- [ ] implement learning rate scheduling
- [x] change learning rate hyperparameter, and other hparams in training
- [ ] implement data augmentation


