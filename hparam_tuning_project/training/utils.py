from hparam_tuning_project.training import Trainer
from hparam_tuning_project.data import PytorchDataset
from hparam_tuning_project.utils import initialize_callbacks
import pytorch_lightning as pl


def build_modules(cfg, extra_callbacks=None):

    if extra_callbacks is None:
        extra_callbacks = []
    dataset = PytorchDataset(**cfg['data_cfg'])
    model = Trainer(**cfg)
    callbacks = initialize_callbacks(cfg['callbacks'])
    callbacks = callbacks + extra_callbacks
    learner = pl.Trainer(**cfg['flags'], callbacks=callbacks)

    return learner, model, dataset


def build_and_fit_modules(cfg, extra_callbacks=None):

    learner, model, dataset = build_modules(cfg, extra_callbacks)
    learner.fit(model, dataset)