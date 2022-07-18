import pytorch_lightning as pl
from torchvision import datasets as ds 
from hparam_tuning_project.models.simple_models import FFN, CNN1
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
import torch

class Trainer(pl.LightningModule):
    """Trainer for the hyperparameter tuning project"""

    model_registry = {
        'FFN': FFN,
        'CNN1': CNN1,
    }

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

    def forward(self):
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

    def configure_optimizers(self):
        return self.optimizer_registry[self.optimizer_cfg['optimizer_id']](self.model.parameters(),
            **self.optimizer_cfg['args'])

    def build_model(self):
       return self.model_registry[self.model_cfg['model_id']](**self.model_cfg['args'])

    def build_loss(self):
        return self.loss_registry[self.loss_cfg['loss_id']](**self.loss_cfg['args'])