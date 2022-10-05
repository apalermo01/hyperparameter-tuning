import pytorch_lightning as pl
from .plot_metrics import PlotMetricsCallback

callback_registry = {
    'early_stop': pl.callbacks.early_stopping.EarlyStopping,
    'plot_metrics': PlotMetricsCallback,
}
