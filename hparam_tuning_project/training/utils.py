from hparam_tuning_project.training import LightningTrainer
from hparam_tuning_project.data import PytorchDataset
from hparam_tuning_project.utils import initialize_callbacks
import lightning.pytorch as pl


def build_modules(cfg, extra_callbacks=None, plugins=None, strategy=None):
    print("pl version = ", pl.__version__)
    if extra_callbacks is None:
        extra_callbacks = []
    if plugins is None:
        plugins = []
    dataset = PytorchDataset(**cfg['data_cfg'])
    model = LightningTrainer(**cfg)
    print("model is LightningModule?", isinstance(model, pl.LightningModule))
    callbacks = initialize_callbacks(cfg['callbacks'])
    callbacks = []
    callbacks = callbacks + extra_callbacks
    learner = pl.Trainer(**cfg['flags'],
                         callbacks=callbacks,
                         plugins=plugins,
                         strategy=strategy)

    return learner, model, dataset


def build_and_fit_modules(cfg, extra_callbacks=None):

    learner, model, dataset = build_modules(cfg, extra_callbacks)
    learner.fit(model, dataset)
