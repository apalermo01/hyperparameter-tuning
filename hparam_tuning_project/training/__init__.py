from torch import optim
from torch.optim import lr_scheduler
from torch.nn import BCEWithLogitsLoss


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
