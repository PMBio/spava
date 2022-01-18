import contextlib
from pprint import pprint

import numpy as np
import optuna
import pyro
import pyro.distributions
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import autograd
from torch import nn
from torch.utils.data import DataLoader, Subset

from data2 import PerturbedCellDataset, file_path


def training_boilerplate(
    trial: optuna.trial.Trial,
    extra_callbacks,
    max_epochs: int,
    log_every_n_steps: int,
    val_check_interval: int,
    model_name: str,
    gpus: int = 1,
):
    logger = TensorBoardLogger(save_dir=file_path("checkpoints"), name=model_name)
    print(f"logging in {logger.experiment.log_dir}")
    version = int(logger.experiment.log_dir.split("version_")[-1])
    trial.set_user_attr("version", version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=file_path(f"{logger.experiment.log_dir}/checkpoints"),
        monitor="batch_val_elbo",
        # every_n_train_steps=2,
        save_last=True,
        save_top_k=3,
    )
    early_stop_callback = EarlyStopping(
        monitor="batch_val_elbo",
        min_delta=0.0001,
        patience=2,
        verbose=True,
        mode="min",
        check_finite=True,
    )
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            PyTorchLightningPruningCallback(trial, monitor="batch_val_elbo"),
            *extra_callbacks,
        ],
        logger=logger,
        num_sanity_val_steps=0,  # track_grad_norm=2,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
    )
    return trainer, logger


# sqlite-backed optuna storage does support nan https://github.com/optuna/optuna/issues/2809
def optuna_nan_workaround(loss):
    # from torch 1.9.0
    # loss = torch.nan_to_num(loss, nan=torch.finfo(loss.dtype).max)
    loss[torch.isnan(loss)] = torch.finfo(loss.dtype).max
    return loss
