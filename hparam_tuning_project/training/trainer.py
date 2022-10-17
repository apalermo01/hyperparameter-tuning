import pytorch_lightning as pl
from torchvision import datasets as ds
from hparam_tuning_project.models.simple_models import FFN, CNN1

import torch.nn.functional as F
import torch

from ..models import model_registry
from . import optimizer_registry, scheduler_registry, loss_registry


class Trainer(pl.LightningModule):
    """Trainer for the hyperparameter tuning project"""

    def __init__(self,
                 model_cfg,
                 optimizer_cfg,
                 data_cfg,
                 loss_cfg,
                 ):
        super(Trainer, self).__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.data_cfg = data_cfg
        self.loss_cfg = loss_cfg

        self.model = self.build_model()
        self.loss = self.build_loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self.model(x)
        target = F.one_hot(target, 10).type(torch.float32)
        loss = self.loss(pred, target)
        self.log('train_loss', loss, on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self.model(x)
        target = F.one_hot(target, 10).type(torch.float32)
        loss = self.loss(pred, target)
        self.log('val_loss', loss, on_step=True, prog_bar=True)
        return {'val_loss': loss}

    def configure_optimizers(self):
        if self.optimizer_cfg['args'] is not None:
            args = self.optimizer_cfg['args']
        else:
            args = dict()
        return optimizer_registry[self.optimizer_cfg['optimizer_id']](self.model.parameters(), **args)

    def build_model(self):
        if self.model_cfg['args'] is not None:
            args = self.model_cfg['args']
        else:
            args = dict()

        return model_registry[self.model_cfg['model_id']](**args)

    def build_loss(self):
        if self.loss_cfg['args'] is not None:
            args = self.loss_cfg['args']
        else:
            args = dict()
        return loss_registry[self.loss_cfg['loss_id']](**args)
