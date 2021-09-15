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



def training_boilerplate(trial: optuna.trial.Trial, extra_callbacks, train_loader, val_loader,
                         train_loader_batch, max_epochs: int, log_every_n_steps: int, val_check_interval: int):
    logger = TensorBoardLogger(save_dir=file_path("checkpoints"), name="expression_vae")
    print(f"logging in {logger.experiment.log_dir}")
    version = int(logger.experiment.log_dir.split("version_")[-1])
    trial.set_user_attr("version", version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=file_path(f"{logger.experiment.log_dir}/checkpoints"),
        monitor="elbo",
        # every_n_train_steps=2,
        save_last=True,
        save_top_k=3,
    )
    early_stop_callback = EarlyStopping(
        monitor="elbo",
        min_delta=0.0001,
        patience=3,
        verbose=True,
        mode="max",
        check_finite=True,
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            PyTorchLightningPruningCallback(trial, monitor="elbo"),
        ],
        logger=logger,
        num_sanity_val_steps=0,  # track_grad_norm=2,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
    )
    return trainer, logger
