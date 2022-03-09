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
    ppp.NUM_WORKERS = 0
    ppp.DETECT_ANOMALY = True
else:
    if "DEBUG_TORCH" in ppp.__dict__:
        ppp.NUM_WORKERS = 0
    else:
        ppp.NUM_WORKERS = 16
    ppp.DETECT_ANOMALY = False

# ppp.NOISE_MODEL = 'gaussian'
# ppp.NOISE_MODEL = 'nb'


import contextlib

import numpy as np
import optuna
import pyro
import pyro.distributions
import pytorch_lightning as pl
import torch
from torch import autograd
from torch import nn
from torch.utils.data import DataLoader, Subset
from pprint import pprint

from analyses.image_viz import get_image
from old_code.data2 import PerturbedRGBCells, quantiles_for_normalization, file_path
from old_code.models.ag_resnet_vae import resnet_encoder, resnet_decoder

pl.seed_everything(1234)


class ImageSampler(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        # normalize = ome_normalization()
        # a = pl_module.negative_binomial_p_logit
        # b = pl_module.boosted_sigmoid(a)
        # trainer.logger.experiment.add_scalars(f'negative_binomial_p_logit', {f'channel{i}': a[i] for i in range(len(
        #     a))}, trainer.global_step)
        # trainer.logger.experiment.add_scalars(f'negative_binomial_p', {f'channel{i}': b[i] for i in range(len(b))},
        #                                       trainer.global_step)

        for dataloader_idx in [0, 1]:
            loader = trainer.val_dataloaders[dataloader_idx]
            dataloader_label = "training" if dataloader_idx == 0 else "validation"
            img = get_image(loader, pl_module)
            if img is not None:
                trainer.logger.experiment.add_image(
                    f"reconstruction/{dataloader_label}", img, trainer.global_step
                )

            # trainer.logger.experiment.add_histogram(f'histograms/{dataloader_label}/image{i}/channel'
            #                                         f'{c}/original', original_masked_c[0].flatten(),
            #                                         trainer.global_step)
            # trainer.logger.experiment.add_histogram(f'histograms/{dataloader_label}/image{i}/channel'
            #                                         f'{c}/reconstructed', reconstructed_masked_c[0].flatten(),
            #                                         trainer.global_step)


def get_detect_anomaly_cm():
    if ppp.DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


class VAE(pl.LightningModule):
    def __init__(self, optuna_parameters, n_channels, input_height=32, **kwargs):
        super().__init__()

        self.save_hyperparameters(kwargs)
        self.save_hyperparameters()
        self.optuna_parameters = optuna_parameters
        self.n_channels = n_channels
        self.enc_out_dim = self.optuna_parameters["enc_out_dim"]
        self.latent_dim = self.optuna_parameters["vae_latent_dims"]
        self.encoder = resnet_encoder(
            first_conv=False,
            maxpool1=False,
            n_channels=self.n_channels,
            enc_out_dim=self.enc_out_dim,
        )
        self.decoder = resnet_decoder(
            latent_dim=self.latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False,
            n_channels=self.n_channels,
        )

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.softplus = nn.Softplus()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.optuna_parameters["learning_rate"]
        )

    def get_dist(self, alpha, beta):
        return pyro.distributions.GammaPoisson(alpha, beta)

    def reconstruction_likelihood(self, alpha, beta, x, mask, corrupted_entries):
        dist = self.get_dist(alpha, beta)
        log_pxz = dist.log_prob(x)

        # measure prob of seeing image under p(x|z)
        # log_pxz = dist.log_prob(x)
        if mask is None:
            mask = torch.ones_like(log_pxz)
        log_pxz[corrupted_entries, :, :] = 0.0
        non_corrupted_count = corrupted_entries.logical_not().sum()
        s = (mask * log_pxz).sum(dim=(1, 2, 3)) / non_corrupted_count
        return s

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

    def loss_function(self, x, alpha, beta, mu, std, z, mask, corrupted_entries):
        # reconstruction loss
        cm = get_detect_anomaly_cm()
        with cm:
            recon_loss = self.reconstruction_likelihood(
                alpha, beta, x, mask, corrupted_entries
            )
            # kl
            kl = self.kl_divergence(z, mu, std)
            # elbo
            elbo = self.optuna_parameters["vae_beta"] * kl - recon_loss
            elbo = elbo.mean()
            if torch.isnan(elbo).any():
                print("nan in loss detected!")
            from old_code.models.boilerplate import optuna_nan_workaround

            elbo = optuna_nan_workaround(elbo)
            return elbo, kl, recon_loss

    def forward(self, x, mask):
        cm = get_detect_anomaly_cm()
        with cm:
            x_encoded = self.encoder(x, mask)
            mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

            # sample z from q
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # decoded
            log_alpha, log_beta = self.decoder(z, mask)
            alpha = torch.exp(log_alpha)
            beta = torch.exp(log_beta)
            return alpha, beta, mu, std, z

    def training_step(self, batch, batch_idx):
        assert len(batch) == 4
        expression, x, mask, corrupted_entries = batch
        # encode x to get the mu and variance parameters
        alpha, beta, mu, std, z = self.forward(x, mask)
        elbo, kl, recon_loss = self.loss_function(
            x, alpha, beta, mu, std, z, mask, corrupted_entries
        )

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "reconstruction": recon_loss.mean(),
            }
        )

        # we set the elbo manually to a high value when we find it nan with optuna_nan_workaround(). When this happes
        # let's tell pytorchlightning to stop traning
        if elbo > 1e20:
            self.trainer.should_stop = True

        return elbo

    def validation_step(self, batch, batch_idx, dataloader_idx):
        assert len(batch) == 4
        expression, x, mask, corrupted_entries = batch
        alpha, beta, mu, std, z = self.forward(x, mask)
        elbo, kl, recon_loss = self.loss_function(
            x, alpha, beta, mu, std, z, mask, corrupted_entries
        )

        self.logger.log_hyperparams(params={}, metrics={"hp_metric": elbo})
        d = {"elbo": elbo, "kl": kl.mean(), "reconstruction": recon_loss.mean()}
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

    def on_post_move_to_device(self):
        self.decoder.mask_conv1.weight = self.encoder.mask_conv1.weight
        self.decoder.mask_conv2.weight = self.encoder.mask_conv2.weight
        self.decoder.mask_conv1x1.weight = self.encoder.mask_conv1x1.weight
        print(
            "sharing the weights between encoder and decoder for the convnet operating on the mask"
        )


# from https://medium.com/@adrian.waelchli/3-simple-tricks-that-will-change-the-way-you-debug-pytorch-5c940aa68b03
class LogComputationalGraph(pl.Callback):
    def __init__(self):
        self.already_logged = False

    def on_validation_start(self, trainer: pl.Trainer, pl_module):
        if not trainer.sanity_checking:
            if not self.already_logged:
                return
                # this code causes a TracerWarning
                # self.already_logged = True
                # sample_image = torch.rand((BATCH_SIZE, len(COOL_CHANNELS), 32, 32))
                # pl_module.logger.experiment.add_graph(VAE(), sample_image)


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
        n = ppp.BATCH_SIZE * 20
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
    from old_code.models.boilerplate import training_boilerplate

    trainer, logger = training_boilerplate(
        trial=trial,
        extra_callbacks=[ImageSampler(), LogComputationalGraph()],
        max_epochs=ppp.MAX_EPOCHS,
        log_every_n_steps=15 if not ppp.DEBUG else 1,
        val_check_interval=1 if ppp.DEBUG else 300,
        model_name="resnet_vae",
    )

    ppp.PERTURB = ppp.PERTURB or False
    print(f"ppp.PERTURB = {ppp.PERTURB}")
    train_loader, val_loader, train_loader_batch = get_loaders(
        perturb=ppp.PERTURB, shuffle_train=True, val_subset=True
    )

    # hyperparameters
    vae_latent_dims = trial.suggest_int("vae_latent_dims", 2, 128)
    vae_beta = trial.suggest_float("vae_beta", 1e-8, 1e-1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1, log=True)
    enc_out_dim = trial.suggest_categorical("enc_out_dim", [128, 256])
    optuna_parameters = dict(
        vae_latent_dims=vae_latent_dims,
        vae_beta=vae_beta,
        learning_rate=learning_rate,
        enc_out_dim=enc_out_dim,
    )
    pprint(optuna_parameters)
    trainer.logger.log_hyperparams(optuna_parameters)

    vae = VAE(
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

    elbo = trainer.callback_metrics["batch_val_elbo"].item()
    return elbo


if __name__ == "__main__":
    # alternative: optuna.pruners.NopPruner()
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = "ag_conv_vae_lightning"
    storage = "sqlite:///" + file_path("optuna_ah.sqlite")
    # optuna.delete_study(study_name=study_name, storage=storage)
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
        HOURS = 60 * 60
        study.optimize(objective, n_trials=100, timeout=2 * HOURS)
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        objective(study.best_trial)
