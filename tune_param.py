
import os
import torch
import pytorch_lightning as pl
import math

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from libs.model import LightningMNISTClassifier
from libs.utils import generate_datetime
from libs.tuning import start_tuning


def train_mnist_tune(tuning_config, data_dir=None, num_epochs=10, num_gpus=0):
    # Only Training
    model = LightningMNISTClassifier(tuning_config, data_dir)

    # ===============================================================================
    # Callback
    # ===============================================================================\
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import EarlyStopping

    early_stop_cb = EarlyStopping(monitor='ptl/val_loss',
                                  patience=5,
                                  verbose=True, mode='min')

    ckpt_cb = ModelCheckpoint(tune.get_trial_dir() + '/checkpoints',
                              save_top_k=5,
                              verbose=True,
                              monitor='ptl/val_loss',
                              mode='min',
                              save_last=True,
                              filename='model_{epoch:03d}-{step}'
                              )

    tune_rp_cb = TuneReportCallback(
        {
            "val_loss": "ptl/val_loss",
            "val_accuracy": "ptl/val_accuracy"
        },
        on="validation_end")

    # ===============================================================================
    # Trainer
    # Note: Must set logger as default with
    # ===============================================================================
    trainer = pl.Trainer(
        progress_bar_refresh_rate=0,  # 0 means no print progress
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        callbacks=[ckpt_cb, tune_rp_cb, early_stop_cb])
    trainer.logger._default_hp_metric = False  # hp_metrc must be False
    trainer.fit(model)


if __name__ == '__main__':
    # ===============================================================================
    # Start Process
    # ===============================================================================

    train_config = {
        'data_dir': '/home/congvm/Workspace/evoke/thirdparty/tune/data',
        'num_epochs': 40,
        'num_gpus': 1
    }

    tuning_config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "opt": tune.choice(['adam', 'sgd'])
    }

    log_dir = 'experiments'
    experiment_name = 'tune_mnist_asha_' + generate_datetime()

    metric_columns = ["val_loss", "val_accuracy", "training_iteration"]

    start_tuning(tuning_config=tuning_config,
                 train_config=train_config,
                 training_func=train_mnist_tune,
                 report_metric_columns=metric_columns,
                 monitor_metric='val_loss',
                 monitor_mode='min',
                 log_dir=log_dir,
                 experiment_name=experiment_name,
                 num_epochs=20, num_workers=20)
