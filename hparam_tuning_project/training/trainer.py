import lightning.pytorch as pl
from sklearn.metrics import accuracy_score
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


class Scorer:

    def __call__(self, pred, actual):
        pred = F.sigmoid(pred)
        indices = torch.argmax(pred, dim=1)
        pred = torch.zeros(pred.shape)
        pred[torch.arange(0, indices.shape[0]), indices] = 1
        return accuracy_score(pred, actual)


class LightningTrainer(pl.LightningModule):
    """Trainer for the hyperparameter tuning project"""

    def __init__(self,
                 model_cfg,
                 optimizer_cfg,
                 data_cfg,
                 loss_cfg,
                 scheduler_cfg=None,
                 scorer_cfg=None,
                 flags=None,
                 callbacks=None,
                 meta=None
                 ):
        super(LightningTrainer, self).__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.data_cfg = data_cfg
        self.loss_cfg = loss_cfg
        if scheduler_cfg is None:
            scheduler_cfg = {}
        self.scheduler_cfg = scheduler_cfg
        if scorer_cfg is None:
            scorer_cfg = {}
        self.scorer_cfg = scorer_cfg

        self.model = self.build_model()
        self.loss = self.build_loss()
        self.scorer = self.build_scorer()

        self.lr = self.optimizer_cfg['args']['lr']
        # storing these to self just so they're saved as hyperparams
        self.flags = flags
        self.callbacks = callbacks
        self.meta = meta
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self.model(x)
        target = F.one_hot(target, 10).type(torch.float32)
        loss = self.loss(pred, target)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self.model(x)
        target = F.one_hot(target, 10).type(torch.float32)
        loss = self.loss(pred, target)
        self.log('val_loss', loss, on_step=True, prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, target = batch
        pred = self.model(x)
        target = F.one_hot(target, 10).type(torch.float32)
        score = self.scorer(pred, target)
        self.log('accuracy', score, on_step=True, prog_bar=True)
        return {'accuracy': score}
    
    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        if self.optimizer_cfg['args'] is not None:
            args = self.optimizer_cfg['args']
        else:
            args = dict()
        optimizer = optimizer_registry[self.optimizer_cfg['optimizer_id']](
            self.model.parameters(), **args)

        if len(self.scheduler_cfg) > 0:
            if self.scheduler_cfg['args'] is not None:
                args = self.scheduler_cfg['args']
            else:
                args = dict()
            scheduler = scheduler_registry[self.scheduler_cfg['scheduler_id']](
                optimizer, **args)
            return [[optimizer], [scheduler]]
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

    def build_scorer(self):
        args = self.scorer_cfg.get('args', {})
        return Scorer(**args)
