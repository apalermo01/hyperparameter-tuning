import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import matplotlib
matplotlib.use('Agg')


class PlotMetricsCallback(Callback):

    def __init__(self):
        self.epoch_num = 0  # increments by 1 at start of first epoch
        # self.epochs = []
        self.batch_num = 0
        self.inter_epoch_train_loss = []
        self.inter_epoch_val_loss = []
        self.train_losses = {}
        self.val_losses = {}

    def on_train_epoch_start(self,
                             trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule") -> None:
        self.train_losses[self.epoch_num] = {}
        # return super().on_train_epoch_start(trainer, pl_module)

    def on_train_batch_end(self,
                           trainer: "pl.Trainer",
                           pl_module: "pl.LightningModule",
                           outputs,
                           batch: Any,
                           batch_idx: int,
                           unused: int = 0) -> None:
        if len(trainer.callback_metrics) > 0:
            self.inter_epoch_train_loss.append(
                trainer.callback_metrics['train_loss'].item())
        # return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, unused)

    # def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     # self.train_losses[self.epoch_num] = {
    #     #     'avg': np.nanmean(self.inter_epoch_train_loss),
    #     #     'std': np.nanstd(self.inter_epoch_train_loss)
    #     # }
    #     # print("ON TRAIN EPOCH END")
    #     # self.inter_epoch_train_loss = []
    #     return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_epoch_start(self,
                                  trainer:
                                  "pl.Trainer",
                                  pl_module: "pl.LightningModule") -> None:
        self.val_losses[self.epoch_num] = {}
        # return super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(self,
                                trainer: "pl.Trainer",
                                pl_module: "pl.LightningModule",
                                Outputs: Optional[STEP_OUTPUT],
                                batch: Any,
                                batch_idx: int,
                                # dataloader_idx: int
                                ) -> None:

        if len(trainer.logged_metrics) > 0 and \
                'val_loss_step' in trainer.logged_metrics:
            self.inter_epoch_val_loss.append(
                trainer.logged_metrics['val_loss_step'].item())

    def on_validation_epoch_end(self,
                                trainer: "pl.Trainer",
                                pl_module: "pl.LightningModule") -> None:

        self.train_losses[self.epoch_num] = {
            'avg': np.nanmean(self.inter_epoch_train_loss),
            'std': np.nanstd(self.inter_epoch_train_loss)
        }
        self.inter_epoch_train_loss = []
        self.val_losses[self.epoch_num] = {
            'avg': np.nanmean(self.inter_epoch_val_loss),
            'std': np.nanstd(self.inter_epoch_val_loss)
        }

        self.inter_epoch_val_loss = []

        if self.epoch_num > 0:
            self.make_plots(path=trainer.log_dir)
        self.epoch_num += 1
        # return super().on_validation_epoch_end(trainer, pl_module)

    def make_plots(self, path):

        fig = plt.figure()
        ax = plt.axes()

        # train
        x = list(self.train_losses.keys())
        avgs = [self.train_losses[key]['avg']
                for key in self.train_losses if 'avg' in self.train_losses[key]]
        stds = [self.train_losses[key]['std'] for key in self.train_losses]
        ax.scatter(x, avgs, label='train', c='b')
        ax.errorbar(x=x, y=avgs, yerr=stds, capsize=5, fmt='none', c='b')

        # val
        x = list(self.val_losses.keys())
        avgs = [self.val_losses[key]['avg'] for key in self.val_losses]
        stds = [self.val_losses[key]['std'] for key in self.val_losses]
        ax.scatter(x, avgs, label='val', c='r')
        ax.errorbar(x=x, y=avgs, yerr=stds, capsize=5, fmt='none', c='r')

        # formatting
        ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("train and val losses")

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(
            path, f"loss_img_epoch_{self.epoch_num}.jpg"), img)
