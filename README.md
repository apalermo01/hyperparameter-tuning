# hyperparameter-tuning
Experimenting with hyperparameter tuning on subset of dataset vs. full dataset


# TODO

- [ x ] build a simple feed forward network
- [ x ] build a simple convolutional network
- [ ] implement callback handling
- [ ] train both on mnist through simple training pipeline
- [ ] change learning rate hyperparameter, and other hparams in training
- [ ] implement data augmentation



General flow of 1 experiment:
- select a model and dataset and optimize hyperparameters. Run this sequence on both the full training dataset and a fraction of the training dataset. 
- Compare the results of the two runs - see if there is any difference in WHAT THE BEST MODEL IS.



1) build network A, do hparam optimization of full dataset. best model = model A
2) build network A, do hparam optimization of a fraction of full dataset. best model = model B
3) compare the selected best hyperparameters for model A and model B - how similar are they?
4) take the hyperparameters from model B, train on full dataset - how similar are these results to model A? 