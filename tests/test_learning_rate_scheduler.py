from platform import architecture
import pytest
from hparam_tuning_project.training import Trainer
import pytorch_lightning as pl
from hparam_tuning_project.data.datasets import PytorchDataset


def test_learning_rate_scheduler():
    training_cfg = dict(
        model_cfg=dict(
            model_id='pytorch_classifier',
            args=dict(
                architecture_id='resnet18',
                pretrained=True,
                num_classes=10,
                num_input_channels=1,
            )
        ),

        data_cfg=dict(
            dataset_id='mnist',
            train=True,
            batch_size=128,
            num_workers=8
        ),

        loss_cfg=dict(
            loss_id='bce_with_logits_loss',
            args=dict(),
        ),

        optimizer_cfg=dict(
            optimizer_id='adam',
            args=dict(
                lr=0.001,
            )
        ),

        scheduler_cfg=dict(
            scheduler_id='step_lr',
            args=dict(
                step_size=2,
                gamma=0.1,
            )
        )
    )

    dataset = PytorchDataset(**training_cfg['data_cfg'])
    model = Trainer(**training_cfg)
    learner = pl.Trainer(max_epochs=2)
    learner.fit(model, dataset)


if __name__ == '__main__':
    test_learning_rate_scheduler()
