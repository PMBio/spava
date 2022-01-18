# aaa = True


class Ppp:
    pass


ppp = Ppp()
# ppp.DEBUG = True
ppp.DEBUG = False
if "aaa" in locals():
    ppp.DEBUG_TORCH = "yessss"
    ppp.DEBUG = True
ppp.MAX_EPOCHS = 6
ppp.BATCH_SIZE = 256
ppp.PERTURB = None
if ppp.DEBUG and not "DEBUG_TORCH" in ppp.__dict__:
    ppp.NUM_WORKERS = 16
    # ppp.NUM_WORKERS = 0
    ppp.DETECT_ANOMALY = True
else:
    if "DEBUG_TORCH" in ppp.__dict__:
        ppp.NUM_WORKERS = 0
    else:
        ppp.NUM_WORKERS = 14
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
from torch_geometric.data import DataLoader as GeometricDataLoader

# from torch.utils.data import Subset
from torch_geometric.nn.conv import GINEConv

from data2 import quantiles_for_normalization, file_path
from graphs import CellExpressionGraphOptimized

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

m = __name__ == "__main__"

# heavy to load, when debugging better to have this done only once so I can call objective (which calls get_loaders,
# which needs these datasets), multiple times
if m:
    _train_ds = None
    _val_ds = None
    _train_ds_perturbed = None
    _val_ds_perturbed = None


def get_ds(split: str, perturbed: bool):
    def _get_ds():
        return CellExpressionGraphOptimized(split, "gaussian", perturbed)

    if split == "train":
        if perturbed:
            global _train_ds_perturbed
            if _train_ds_perturbed is None:
                _train_ds_perturbed = _get_ds()
            return _train_ds_perturbed
        else:
            global _train_ds
            if _train_ds is None:
                _train_ds = _get_ds()
            return _train_ds
    elif split == "validation":
        if perturbed:
            global _val_ds_perturbed
            if _val_ds_perturbed is None:
                _val_ds_perturbed = _get_ds()
            return _val_ds_perturbed
        else:
            global _val_ds
            if _val_ds is None:
                _val_ds = _get_ds()
            return _val_ds
    else:
        raise NotImplementedError()


##


def get_detect_anomaly_cm():
    if ppp.DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


class GnnEncoder(nn.Module):
    def __init__(self, in_channels, optuna_parameters=None):
        super().__init__()
        self.in_channels = in_channels
        if optuna_parameters is not None:
            self.p_dropout = output["p_dropout"]
        else:
            self.p_dropout = 0.1
        gine0_nn = nn.Sequential(
            nn.Linear(self.in_channels + 1, self.in_channels + 1),
            nn.BatchNorm1d(self.in_channels + 1),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
            nn.Linear(self.in_channels + 1, self.in_channels),
        )
        gine1_nn = nn.Sequential(
            nn.Linear(self.in_channels + 1, self.in_channels + 1),
            nn.BatchNorm1d(self.in_channels + 1),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
            nn.Linear(self.in_channels + 1, self.in_channels),
        )
        self.gcn0 = GINEConv(gine0_nn)
        self.gcn1 = GINEConv(gine1_nn)
        self.linear0 = nn.Linear(1, self.in_channels + 1)
        self.linear1 = nn.Linear(self.in_channels + 1, self.in_channels + 1)
        self.batch_norm0 = nn.BatchNorm1d(self.in_channels)
        self.dropout0 = nn.Dropout(p=self.p_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, is_center):
        original_x = x
        e = self.linear1(self.relu(self.linear0(edge_attr)))
        x_with_center_info = torch.cat((x, is_center.view(-1, 1)), dim=1)
        x = self.gcn0(x_with_center_info, edge_index, e) + original_x
        x = self.batch_norm0(x)
        x = self.relu(x)
        x = self.dropout0(x)
        x_with_center_info = torch.cat((x, is_center.view(-1, 1)), dim=1)
        x = self.gcn1(x_with_center_info, edge_index, e) + original_x
        (indices_is_center,) = torch.where(is_center == 1)
        x = x[indices_is_center, :]
        # x = is_center @ x
        # n = torch.argmax(is_center, dim=0)
        # x = x[n, :]
        return x


##
if m and False:
    ##
    ds = CellExpressionGraphOptimized("validation", "gaussian")
    loader = GeometricDataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    ##
    model = GnnEncoder(in_channels=39)
    data = loader.__iter__().__next__()
    output = model(data.x, data.edge_index, data.edge_attr, data.is_center)
    ##
    import sys

    sys.exit(0)


##
class GnnVae(pl.LightningModule):
    def __init__(self, optuna_parameters, n_channels, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters()
        self.optuna_parameters = optuna_parameters
        self.n_channels = n_channels
        self.latent_dim = self.optuna_parameters["vae_latent_dims"]
        self.out_channels = self.n_channels
        self.gnn_encoder = GnnEncoder(in_channels=self.n_channels)
        self.linear0 = nn.Linear(self.n_channels, 20)
        self.dropout0 = nn.Dropout(p=optuna_parameters["p_dropout"])
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

    def encoder(self, x, edge_index, edge_attr, is_center):
        y = F.relu(self.gnn_encoder(x, edge_index, edge_attr, is_center))
        y = self.dropout0(y)
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

    def forward(self, x, edge_index, edge_attr, is_center):
        cm = get_detect_anomaly_cm()
        with cm:
            mu, log_var = self.encoder(x, edge_index, edge_attr, is_center)

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

    def unfold_batch(self, batch):
        # Batch(batch=[2862], edge_attr=[49886, 1], edge_index=[2, 49886], is_center=[2862], is_perturbed=[2862, 39],
        #       ptr=[33], x=[2862, 39])
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        is_center = batch.is_center
        # the scalar product will not introduce numbers not in {0., 1.}
        # corrupted_entries = (is_center @ batch.is_perturbed.float()).bool()
        corrupted_entries = batch.is_perturbed[torch.where(is_center == 1.0)[0], :]
        expression = x[torch.where(is_center == 1.0)[0], :]
        # expression = is_center @ x
        return x, edge_index, edge_attr, is_center, corrupted_entries, expression

    def training_step(self, batch, batch_idx):
        (
            x,
            edge_index,
            edge_attr,
            is_center,
            corrupted_entries,
            expression,
        ) = self.unfold_batch(batch)
        a, mu, std, z = self.forward(x, edge_index, edge_attr, is_center)
        # n = torch.argmax(is_center, dim=0)
        # expression = x[n, :]
        expression = x[torch.where(is_center == 1.0)[0], :]
        # expression = is_center @ x
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
        if not self.trainer.should_stop:
            (
                x,
                edge_index,
                edge_attr,
                is_center,
                corrupted_entries,
                expression,
            ) = self.unfold_batch(batch)
            a, mu, std, z = self.forward(x, edge_index, edge_attr, is_center)
            n = torch.argmax(is_center, dim=0)
            expression = x[n, :]
            elbo, kl, recon_loss = self.loss_function(
                expression, a, mu, std, z, corrupted_entries
            )
        else:
            from models.boilerplate import optuna_nan_workaround

            nan = torch.Tensor([float("NaN")]).to(self.device)
            elbo = optuna_nan_workaround(nan)
            kl = recon_loss = elbo

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


if m and False:
    #
    ds = CellExpressionGraphOptimized("validation", "gaussian")
    loader = GeometricDataLoader(
        ds, batch_size=32, shuffle=True, num_workers=ppp.NUM_WORKERS
    )
    #
    o = dict(vae_latent_dims=10, log_c=0)
    model = GnnVae(optuna_parameters=o, n_channels=39)
    data = loader.__iter__().__next__()
    output = model(data.x, data.edge_index, data.edge_attr, data.is_center)
    a, mu, std, z = output
    print(
        f"a.shape = {a.shape}, mu.shape = {mu.shape}, std.shape = {std.shape}, z.shape = {z.shape}"
    )
    # ##
    # import sys
    #
    # sys.exit(0)

import torch.utils.data
import torch_geometric.data


class GeometricSubset(torch_geometric.data.Dataset):
    def __init__(self, ds, subset_length: int):
        super().__init__()
        self.subset_indices = np.random.choice(len(ds), subset_length, replace=False)
        self.ds = ds

    def __len__(self):
        return len(self.subset_indices)

    def get(self, item):
        return self.ds[self.subset_indices[item].item()]


def get_loaders(
    perturb: bool,
    shuffle_train=False,
    val_subset=False,
):
    train_ds = get_ds("train", perturb)
    val_ds = get_ds("validation", perturb)
    assert train_ds is not None
    assert val_ds is not None

    if ppp.DEBUG:
        n = ppp.BATCH_SIZE * 2
    else:
        n = ppp.BATCH_SIZE * 10
    # torch.data.utils.Subset don't work with torch_geometric
    # indices = torch.from_numpy(np.random.choice(len(train_ds), n, replace=False))
    # train_subset = Subset(train_ds, indices)
    train_subset = GeometricSubset(train_ds, n)

    if ppp.DEBUG:
        d = train_subset
    else:
        d = train_ds
    train_loader = GeometricDataLoader(
        d,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
        shuffle=shuffle_train,
    )
    # train_loader.__iter__().__next__()
    train_loader_batch = GeometricDataLoader(
        train_subset,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
    )

    # torch.data.utils.Subset don't work with torch_geometric
    # the val set is a bit too big for training a lot of image models, we are fine with evaluating the generalization
    # on a subset of the data
    if val_subset:
        #     indices = torch.from_numpy(np.random.choice(len(val_ds), n, replace=False))
        #     subset = Subset(val_ds, indices)
        subset = GeometricSubset(val_ds, n)
    else:
        subset = val_ds
    val_loader = GeometricDataLoader(
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
        log_every_n_steps=150 if not ppp.DEBUG else 1,
        val_check_interval=1 if ppp.DEBUG else 800,
        model_name="gnn_vae",
        gpus=1,
    )

    ppp.PERTURB = ppp.PERTURB or False
    print(f"ppp.PERTURB = {ppp.PERTURB}")
    train_loader, val_loader, train_loader_batch = get_loaders(
        perturb=ppp.PERTURB, shuffle_train=True, val_subset=True
    )

    # hyperparameters
    vae_latent_dims = 10
    vae_beta = trial.suggest_float("vae_beta", 1e-8, 1e-1, log=True)
    log_c = trial.suggest_float("log_c", -3, 3)
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-1, log=True)
    p_dropout = trial.suggest_float("p_dropout", 0.0, 0.5)
    optuna_parameters = dict(
        vae_latent_dims=vae_latent_dims,
        vae_beta=vae_beta,
        log_c=log_c,
        learning_rate=learning_rate,
        p_dropout=p_dropout,
    )
    pprint(optuna_parameters)
    trainer.logger.log_hyperparams(optuna_parameters)

    vae = GnnVae(
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
    study_name = "ai_gnn_vae"
    storage = "sqlite:///" + file_path("optuna_aj.sqlite")
    # optuna.delete_study(study_name=study_name, storage=storage)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
        study_name=study_name,
    )
    TRAIN_PERTURBED = True
    # TRAIN_PERTURBED = False
    if not TRAIN_PERTURBED:
        HOURS = 60 * 60
        study.optimize(objective, n_trials=100, timeout=8 * HOURS)
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
        print(
            f"(best) trial.number = {trial.number}, (best) trial._user_attrs = {trial._user_attrs}"
        )
        # import pandas as pd
        # pd.set_option('expand_frame_repr', False)
        # df = study.trials_dataframe()
        # print(df.sort_values(by='value'))
        objective(trial)

# best trial: 0 (optuna), 109 (tensorboard)
# corresponding perturbed: 152 (tensorboard)
