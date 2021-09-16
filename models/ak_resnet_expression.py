# aaa = True


class Ppp:
    pass


ppp = Ppp()
if "aaa" in locals():
    ppp.DEBUG_TORCH = "yessss"
ppp.MAX_EPOCHS = 6
ppp.BATCH_SIZE = 1024
ppp.PERTURB = None
# ppp.DEBUG = True
ppp.DEBUG = False
if ppp.DEBUG and not "DEBUG_TORCH" in ppp.__dict__:
    ppp.NUM_WORKERS = 16
    # ppp.NUM_WORKERS = 0
    ppp.DETECT_ANOMALY = True
else:
    if "DEBUG_TORCH" in ppp.__dict__:
        ppp.NUM_WORKERS = 0
    else:
        ppp.NUM_WORKERS = 16
    ppp.DETECT_ANOMALY = False

import contextlib
import os
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
from torch.utils.data import DataLoader, Subset

from data2 import PerturbedRGBCells, quantiles_for_normalization, file_path
from models.ag_resnet_vae import resnet_encoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_detect_anomaly_cm():
    if ppp.DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


class ResNetToExpression(pl.LightningModule):
    def __init__(self, optuna_parameters, n_channels, input_height=32, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters()
        self.optuna_parameters = optuna_parameters
        self.n_channels = n_channels
        self.latent_dim = self.optuna_parameters["vae_latent_dims"]
        self.out_channels = self.n_channels
        self.enc_out_dim = self.optuna_parameters["enc_out_dim"]
        self.resnet_encoder = resnet_encoder(
            first_conv=False,
            maxpool1=False,
            n_channels=n_channels,
            enc_out_dim=self.enc_out_dim,
        )
        self.linear0 = nn.Linear(self.enc_out_dim, 20)
        self.linear1_0 = nn.Linear(20, self.latent_dim)
        self.linear1_1 = nn.Linear(20, self.latent_dim)

        # decoder stuff
        self.decoder0 = nn.Linear(self.latent_dim, 15)
        self.decoder1 = nn.Linear(15, 20)
        self.decoder2 = nn.Linear(20, 30)
        self.decoder3_a = nn.Linear(30, self.out_channels)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

        self.log_c = nn.Parameter(torch.Tensor([self.optuna_parameters["log_c"]] * 39))

    def encoder(self, x, mask):
        y = F.relu(self.resnet_encoder(x, mask))
        y = F.relu(self.linear0(y))
        mu = self.linear1_0(y)
        log_var = self.linear1_1(y)
        return mu, log_var

    def decoder(self, z):
        z = F.relu(self.decoder0(z))
        z = F.relu(self.decoder1(z))
        z = F.relu(self.decoder2(z))
        decoded_a = self.decoder3_a(z)
        return decoded_a

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.optuna_parameters["learning_rate"]
        )

    def reconstruction_likelihood(self, a, x, corrupted_entries):
        dist = self.get_dist(a)
        # measure prob of seeing image under p(x|z)
        # bad variable naming :P
        zero = torch.tensor([2.0]).to(a.device)
        if torch.any(dist.log_prob(zero).isinf()):
            print("infinite value detected")
            self.trainer.should_stop = True
        if torch.any(dist.log_prob(zero).isnan()):
            print("nan value detected")
            self.trainer.should_stop = True
        log_pxz = dist.log_prob(x)
        log_pxz[corrupted_entries] = 0.0
        non_corrupted_count = corrupted_entries.logical_not().sum()
        s = log_pxz.sum(dim=-1) / non_corrupted_count
        return s

    def get_dist(self, a):
        return pyro.distributions.Normal(a, torch.exp(self.log_c))

    def expected_value(self, a):
        dist = self.get_dist(a)
        return dist.mean

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def loss_function(self, expression, a, mu, std, z, corrupted_entries):
        cm = get_detect_anomaly_cm()
        with cm:
            recon_loss = self.reconstruction_likelihood(
                a, expression, corrupted_entries
            )
            kl = self.kl_divergence(z, mu, std)
            elbo = self.optuna_parameters["vae_beta"] * kl - recon_loss
            elbo = elbo.mean()
            if torch.isnan(elbo).any():
                print("nan in loss detected!")
                self.trainer.should_stop = True
            if torch.isinf(elbo).any():
                print("inf in loss detected!")
                self.trainer.should_stop = True
            from models.boilerplate import optuna_nan_workaround

            elbo = optuna_nan_workaround(elbo)
            return elbo, kl, recon_loss

    def forward(self, x, mask):
        x_and_mask = torch.cat((x, mask), dim=1)
        cm = get_detect_anomaly_cm()
        with cm:
            mu, log_var = self.encoder(x, mask)

            # sample z from q
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # decoded
            a = self.decoder(z)
            if (
                    torch.isnan(a).any()
                    or torch.isnan(mu).any()
                    or torch.isnan(std).any()
                    or torch.isnan(z).any()
            ):
                print("nan in forward detected!")
                self.trainer.should_stop = True

            return a, mu, std, z

    def training_step(self, batch, batch_idx):
        assert len(batch) == 4
        expression, x, mask, corrupted_entries = batch
        assert len(x.shape) == 4
        assert x.shape[1] == 39
        assert len(expression.shape) == 2
        assert expression.shape[1] == 39
        a, mu, std, z = self.forward(x, mask)
        elbo, kl, recon_loss = self.loss_function(
            expression, a, mu, std, z, corrupted_entries
        )

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "reconstruction": recon_loss.mean(),
            }
        )

        return elbo

    def validation_step(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 4
        expression, x, mask, corrupted_entries = batch
        a, mu, std, z = self.forward(x, mask)
        elbo, kl, recon_loss = self.loss_function(
            expression, a, mu, std, z, corrupted_entries
        )

        self.logger.log_hyperparams(params={}, metrics={"hp_metric": elbo})
        d = {
            "elbo": elbo,
            "kl": kl.mean(),
            "reconstruction": recon_loss.mean(),
        }
        return d

    def validation_epoch_end(self, outputs):
        if not self.trainer.sanity_checking:
            assert type(outputs) is list
            batch_val_elbo = None
            for i, o in enumerate(outputs):
                for k in ["elbo", "kl", "reconstruction"]:
                    avg_loss = torch.stack([x[k] for x in o]).mean().cpu().detach()
                    phase = "training" if i == 0 else "validation"
                    self.logger.experiment.add_scalar(
                        f"avg_metric/{k}/{phase}", avg_loss, self.global_step
                    )
                    if phase == "validation" and k == "elbo":
                        batch_val_elbo = avg_loss
            assert batch_val_elbo is not None
            self.log("batch_val_elbo", batch_val_elbo)


def get_loaders(
        perturb: bool,
        shuffle_train=False,
        val_subset=False,
):
    train_ds = PerturbedRGBCells("train", augment=True, aggressive_rotation=True)
    train_ds_validation = PerturbedRGBCells("train")
    val_ds = PerturbedRGBCells("validation")

    if perturb:
        train_ds.perturb()
        train_ds_validation.perturb()
        val_ds.perturb()

    if ppp.DEBUG:
        n = ppp.BATCH_SIZE * 2
    else:
        n = ppp.BATCH_SIZE * 40
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

    # the val set is a bit too big for training a lot of image models, we are fine with evaluating the generalization
    # on a subset of the data
    if val_subset:
        indices = np.random.choice(len(val_ds), n, replace=False)
        subset = Subset(val_ds, indices)
    else:
        subset = val_ds
    val_loader = DataLoader(
        subset,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, train_loader_batch


def objective(trial: optuna.trial.Trial) -> float:
    from models.boilerplate import training_boilerplate

    trainer, logger = training_boilerplate(
        trial=trial,
        extra_callbacks=[],
        max_epochs=ppp.MAX_EPOCHS,
        log_every_n_steps=15 if not ppp.DEBUG else 1,
        val_check_interval=1 if ppp.DEBUG else 300,
        model_name="image_to_expression",
    )

    ppp.PERTURB = ppp.PERTURB or False
    print(f"ppp.PERTURB = {ppp.PERTURB}")
    train_loader, val_loader, train_loader_batch = get_loaders(
        perturb=ppp.PERTURB, shuffle_train=True, val_subset=True
    )

    # hyperparameters
    vae_latent_dims = 10
    # vae_latent_dims = trial.suggest_int("vae_latent_dims", 5, 10)
    vae_beta = trial.suggest_float("vae_beta", 1e-8, 1e-1, log=True)
    log_c = trial.suggest_float("log_c", -3, 3)
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1, log=True)
    enc_out_dim = trial.suggest_categorical("enc_out_dim", [128, 256])
    optuna_parameters = dict(
        vae_latent_dims=vae_latent_dims,
        vae_beta=vae_beta,
        log_c=log_c,
        learning_rate=learning_rate,
        enc_out_dim=enc_out_dim,
    )
    pprint(optuna_parameters)
    trainer.logger.log_hyperparams(optuna_parameters)

    vae = ResNetToExpression(
        optuna_parameters=optuna_parameters,
        n_channels=len(quantiles_for_normalization),
        **ppp.__dict__,
    )
    trainer.fit(
        vae,
        train_dataloaders=train_loader,
        val_dataloaders=[train_loader_batch, val_loader],
    )
    print(f"finished logging in {logger.experiment.log_dir}")

    print(trainer.callback_metrics)
    elbo = trainer.callback_metrics["batch_val_elbo"].item()
    return elbo


if __name__ == "__main__":
    # alternative: optuna.pruners.NopPruner()
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = "ak_resnet_expression"
    storage = "sqlite:///" + file_path("optuna_aj.sqlite")
    # optuna.delete_study(study_name=study_name, storage=storage)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
        study_name=study_name,
    )
    # TRAIN_PERTURBED = True
    TRAIN_PERTURBED = False
    if not TRAIN_PERTURBED:
        HOURS = 60 * 60
        study.optimize(objective, n_trials=100, timeout=6 * HOURS)
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        ppp.PERTURB = True
        trial = study.best_trial
        objective(trial)
