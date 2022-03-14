##
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
import colorama

from analyses.torch_boilerplate import (
    optuna_nan_workaround,
    ZeroInflatedGamma,
    ZeroInflatedNormal,
)

from models.expression_vae import VAE


class Ppp:
    pass


ppp = Ppp()

ppp.LOG_PER_CHANNEL_VALUES = False
if "SPATIALMUON_TEST" not in os.environ:
    ppp.MAX_EPOCHS = 15
else:
    ppp.MAX_EPOCHS = 2
ppp.BATCH_SIZE = 256
ppp.MONTE_CARLO = True
ppp.MASK_LOSS = True
ppp.PERTURB = None
# ppp.DEBUG = True
ppp.DEBUG = False
if ppp.DEBUG:
    ppp.NUM_WORKERS = 0
    ppp.DETECT_ANOMALY = True
else:
    ppp.NUM_WORKERS = 10
    # ppp.NUM_WORKERS = 0
    ppp.DETECT_ANOMALY = False
# ppp.NOISE_MODEL = "gaussian"
# ppp.NOISE_MODEL = 'gamma'
# ppp.NOISE_MODEL = 'zip'
# ppp.NOISE_MODEL = 'zin'
# ppp.NOISE_MODEL = 'log_normal'
# ppp.NOISE_MODEL = 'zi_gamma'
# ppp.NOISE_MODEL = 'nb'
ppp.NOISE_MODEL = "zinb"

import contextlib
from pprint import pprint

import numpy as np
import optuna
import pyro
import pyro.distributions
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import autograd
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm.auto import tqdm
import copy

from datasets.loaders.visium_mousebrain_loaders import CellsDataset

pl.seed_everything(1234)

from utils import file_path, get_execute_function
from datasets.visium_mousebrain import get_smu_file

s = get_smu_file(read_only=True)
n_channels = len(s["visium"]["processed"].var)
s.backing.close()

e_ = get_execute_function()


def get_loaders(
    perturb: bool = False,
    shuffle_train=False,
):
    train_ds = CellsDataset("train", only_expression=True, raw_counts=True)
    train_ds_validation = CellsDataset("train", only_expression=True, raw_counts=True)
    val_ds = CellsDataset("validation", only_expression=True, raw_counts=True)
    if perturb:
        train_ds = train_ds.perturb()
        train_ds_validation = train_ds_validation.perturb()
        val_ds = val_ds.perturb()

    # print(
    #     f"len(train_ds) = {len(train_ds[0])}, train_ds[0] = {train_ds[0][0]}, train_ds[0].shape ="
    #     f" {train_ds[0][0].shape}"
    # )

    if ppp.DEBUG:
        n = ppp.BATCH_SIZE * 2
    else:
        n = ppp.BATCH_SIZE * 20
    # when testing we otherwise have n > len(train_ds)
    n = min(n, len(train_ds))
    indices = np.random.choice(len(train_ds), n, replace=False)
    train_subset = Subset(train_ds_validation, indices)

    if ppp.DEBUG:
        d = train_subset
    else:
        d = train_ds
    train_loader = DataLoader(
        d,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
        shuffle=shuffle_train,
    )
    train_loader_batch = DataLoader(
        train_subset,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
    )

    # indices = np.random.choice(len(val_ds), n, replace=False)
    # val_subset = Subset(val_ds, indices)
    val_subset = val_ds
    val_loader = DataLoader(
        val_subset,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, train_loader_batch

    # class MySampler(Sampler):
    #     def __init__(self, my_ordered_indices):
    #         self.my_ordered_indices = my_ordered_indices
    #
    #     def __iter__(self):
    #         return self.my_ordered_indices.__iter__()
    #
    #     def __len__(self):
    #         return len(self.my_ordered_indices)
    #
    # faulty_epoch = 181
    # l = list(range(len(train_ds)))
    # indices = l[n:] + l[:n]
    # indices = indices[:10]
    # debug_sampler = MySampler(indices)

    # debug_train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
    #                                 sampler=debug_sampler)


def objective(trial: optuna.trial.Trial) -> float:
    from analyses.torch_boilerplate import training_boilerplate

    from models.expression_vae import ImageSampler, LogComputationalGraph

    val_check_internal = 1 if ppp.DEBUG or "SPATIALMUON_TEST" in os.environ else 10
    if "MAX_EPOCHS" in trial.user_attrs:
        ppp.MAX_EPOCHS = trial.user_attrs["MAX_EPOCHS"]
        print(
            f"{colorama.Fore.MAGENTA}found attribute in trial.user_attrs{colorama.Fore.RESET}"
        )
    print(f"ppp.MAX_EPOCHS = {ppp.MAX_EPOCHS}")
    trainer, logger = training_boilerplate(
        trial=trial,
        extra_callbacks=[ImageSampler(), LogComputationalGraph()],
        max_epochs=ppp.MAX_EPOCHS,
        log_every_n_steps=10 if not ppp.DEBUG else 1,
        val_check_interval=val_check_internal,
        model_name="visium_mousebrain_expression_vae",
        early_stopping_patience=4,
        early_stopping_min_delta=1e-5,
    )

    # hyperparameters
    vae_latent_dims = trial.suggest_int("vae_latent_dims", 8, 20)
    vae_beta = trial.suggest_float("vae_beta", 1e-8, 1e-1, log=True)
    log_c = trial.suggest_float("log_c", -3, 3)
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e1, log=True)
    dropout_alpha = trial.suggest_float("dropout_alpah", 0.0, 0.2)

    # # leading to nan values
    # vae_latent_dims = 5
    # vae_beta = 1.231490907826831e-7
    # log_c = -2.20108960756478
    # learning_rate = 0.5985518082388829

    optuna_parameters = dict(
        vae_latent_dims=vae_latent_dims,
        vae_beta=vae_beta,
        log_c=log_c,
        learning_rate=learning_rate,
        dropout_alpha=dropout_alpha,
    )
    pprint(optuna_parameters)
    trainer.logger.log_hyperparams(optuna_parameters)

    vae = VAE(
        optuna_parameters=optuna_parameters,
        in_channels=n_channels,
        mask_loss=ppp.MASK_LOSS,
        **ppp.__dict__,
    )

    ppp.PERTURB = ppp.PERTURB or False
    print(f"ppp.PERTURB = {ppp.PERTURB}")
    train_loader, val_loader, train_loader_batch = get_loaders(
        shuffle_train=True,
    )
    try:
        trainer.fit(
            vae,
            train_dataloaders=train_loader,
            val_dataloaders=[train_loader_batch, val_loader],
        )
        print(f"finished logging in {logger.experiment.log_dir}")

        elbo = trainer.callback_metrics["batch_val_elbo"].item()
        torch.cuda.empty_cache()
        return elbo
    except RuntimeError as e:
        if str(e) == "manual abort":
            elbo = optuna_nan_workaround(torch.tensor(float("nan")))
            print("manual abort")
            torch.cuda.empty_cache()
            return elbo
        else:
            raise e


if e_() or __name__ == "__main__":
    # alternative: optuna.pruners.NopPruner()
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = "visium_mousebrain_expression"
    if study_name == "visium_mousebrain_expression_perturbed":
        ppp.PERTURB = True
    elif study_name == "visium_mousebrain_expression":
        ppp.PERTURB = False
    else:
        raise NotImplementedError()
    # storage = 'mysql://l989o@optuna'
    storage = "sqlite:///" + file_path("optuna_visium_mousebrain_expression.sqlite")
    # optuna.delete_study(study_name, storage)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
        study_name=study_name,
    )
    OPTIMIZE = True
    # OPTIMIZE = False
    if OPTIMIZE:
        if "SPATIALMUON_TEST" not in os.environ:
            n_trials = 100
        else:
            n_trials = 1
        study.optimize(
            objective, n_trials=n_trials, timeout=4 * 3600, gc_after_trial=True
        )
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        # TRAIN = True
        TRAIN = False
        if TRAIN:
            objective(study.best_trial)
        else:
            print("best trial:")
            print(study.best_trial)

            df = study.trials_dataframe()
            df = df.sort_values(by="value")
            import pandas as pd

            pd.set_option("display.max_rows", 500)
            pd.set_option("display.max_columns", 500)
            pd.set_option("display.width", 1000)
            print(df)
            # df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
