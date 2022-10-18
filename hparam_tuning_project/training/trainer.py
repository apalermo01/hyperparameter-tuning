import pytorch_lightning as pl
from torchvision import datasets as ds
from hparam_tuning_project.models.simple_models import FFN, CNN1

import torch.nn.functional as F
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import BCEWithLogitsLoss

from ..models import model_registry


optimizer_registry = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'sgd': optim.SGD
}


scheduler_registry = {
    'lambda_lr': lr_scheduler.LambdaLR,
    'multiplicative_lr': lr_scheduler.MultiplicativeLR,
    'step_lr': lr_scheduler.StepLR,
    'multi_step_lr': lr_scheduler.MultiStepLR,
    'constant_lr': lr_scheduler.ConstantLR,
    'linear_lr': lr_scheduler.LinearLR,
    'exponential_lr': lr_scheduler.ExponentialLR,
    'cosine_annealing_lr': lr_scheduler.CosineAnnealingLR,
    'reduce_lr_on_plateau': lr_scheduler.ReduceLROnPlateau,
    'cyclic_lr': lr_scheduler.CyclicLR,
    'one_cycle_lr': lr_scheduler.OneCycleLR,
    'cosine_annealing_warm_restarts': lr_scheduler.CosineAnnealingWarmRestarts,
}


loss_registry = {
    'bce_with_logits_loss': BCEWithLogitsLoss
}


class Trainer(pl.LightningModule):
    """Trainer for the hyperparameter tuning project"""

    def __init__(self,
                 model_cfg,
                 optimizer_cfg,
                 data_cfg,
                 loss_cfg,
                 scheduler_cfg=None
                 ):
        super(Trainer, self).__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.data_cfg = data_cfg
        self.loss_cfg = loss_cfg
        self.scheduler_cfg = scheduler_cfg
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
        optimizer = optimizer_registry[self.optimizer_cfg['optimizer_id']](self.model.parameters(), **args)

        if self.scheduler_cfg is not None:
            if self.scheduler_cfg['args'] is not None:
                args = self.scheduler_cfg['args']
            else:
                args = dict()
            scheduler = scheduler_registry[self.scheduler_cfg['scheduler_id']](optimizer, **args)
            return {"optimizer": optimizer, "scheduler": scheduler}
        return optimizer

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
