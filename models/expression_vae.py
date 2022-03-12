##
import contextlib

import pyro
import pyro.distributions
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import autograd
from torch import nn

from analyses.torch_boilerplate import (
    optuna_nan_workaround,
    ZeroInflatedGamma,
    ZeroInflatedNormal,
)

pl.seed_everything(1234)

from utils import get_execute_function
from datasets.imc_data import get_smu_file

e_ = get_execute_function()

class ImageSampler(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        if pl_module.ppp.LOG_PER_CHANNEL_VALUES:
            trainer.logger.experiment.add_scalars(
                "c",
                {
                    f"channel{i}": torch.exp(pl_module.log_c[i])
                    for i in range(len(pl_module.log_c))
                },
                trainer.global_step,
            )
        # trainer.logger.experiment.add_scalars('d', {f'channel{i}': torch.sigmoid(pl_module.logit_d[i]) for i in range(
        #     len(
        #         pl_module.logit_d))}, trainer.global_step)

        # for dataloader_idx in [0, 1]:
        # loader = trainer.val_dataloaders[dataloader_idx]
        # dataloader_label = 'training' if dataloader_idx == 0 else 'validation'
        # trainer.logger.experiment.add_image(f'reconstruction/{dataloader_label}', img,
        #                                     trainer.global_step)

        # trainer.logger.experiment.add_histogram(f'histograms/{dataloader_label}/image{i}/channel'
        #                                         f'{c}/original', original_masked_c[0].flatten(),
        #                                         trainer.global_step)


def get_detect_anomaly_cm(ppp):
    if ppp.DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


class VAE(pl.LightningModule):
    def __init__(
        self,
        optuna_parameters,
        in_channels,
        ppp,
        mask_loss: bool = None,
    ):
        super().__init__()
        self.ppp = ppp
        self.save_hyperparameters(ppp.__dict__)
        self.save_hyperparameters()
        self.optuna_parameters = optuna_parameters
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.latent_dim = self.optuna_parameters["vae_latent_dims"]
        self.mask_loss = mask_loss

        self.encoder0 = nn.Linear(self.in_channels, 30)
        self.encoder1 = nn.Linear(30, 20)
        self.encoder2 = nn.Linear(20, 15)
        self.encoder3_mean = nn.Linear(15, self.latent_dim)
        self.encoder3_log_var = nn.Linear(15, self.latent_dim)
        self.decoder0 = nn.Linear(self.latent_dim, 15)
        self.decoder1 = nn.Linear(15, 20)
        self.decoder2 = nn.Linear(20, 30)
        self.decoder3_a = nn.Linear(30, self.out_channels)
        self.softplus = nn.Softplus()
        self.decoder3_b = nn.Linear(30, self.out_channels)
        self.sigmoid = nn.Sigmoid()

        self.log_c = nn.Parameter(torch.Tensor([self.optuna_parameters["log_c"]] * 39))
        self.logit_d = nn.Parameter(torch.logit(torch.Tensor([0.001] * 39)))

    def encoder(self, x):
        x = F.relu(self.encoder0(x))
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        mean = self.encoder3_mean(x)
        log_var = self.encoder3_log_var(x)
        return mean, log_var

    def decoder(self, z):
        z = F.relu(self.decoder0(z))
        z = F.relu(self.decoder1(z))
        z = F.relu(self.decoder2(z))
        decoded_a = self.decoder3_a(z)
        decoded_b = self.decoder3_b(z)
        if self.ppp.NOISE_MODEL in ["gamma", "zi_gamma", "nb"]:
            decoded_a = self.softplus(decoded_a) + 2
        elif self.ppp.NOISE_MODEL == "zip":
            decoded_a = self.softplus(decoded_a)
        if self.ppp.NOISE_MODEL in ["zip", "zig", "zi_gamma"]:
            decoded_b = self.sigmoid(decoded_b)
        return decoded_a, decoded_b

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.optuna_parameters["learning_rate"]
        )

    @staticmethod
    def kld_loss(mu, log_var):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        # kld = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
        # the mean for dim=0 is done by loss_function()
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return kld

    def mse_loss(self, recon_x, x, corrupted_entries):
        # mse_loss = nn.MSELoss(reduction='mean')
        # err_absolute = mse_loss(recon_x, x)
        se_loss = torch.square(recon_x - x)
        if self.mask_loss:
            se_loss[corrupted_entries] = 0.0
        non_corrupted_count = corrupted_entries.logical_not().sum()
        mse_loss = se_loss.sum(dim=-1) / non_corrupted_count
        return mse_loss

    def get_dist(self, a, b):
        if self.ppp.NOISE_MODEL == "gaussian":
            dist = pyro.distributions.Normal(a, torch.exp(self.log_c))
        elif self.ppp.NOISE_MODEL == "zin":
            dist = ZeroInflatedNormal(a, torch.exp(self.log_c), gate=b)
        elif self.ppp.NOISE_MODEL == "gamma":
            dist = pyro.distributions.Gamma(a, torch.exp(self.log_c))
        elif self.ppp.NOISE_MODEL == "zi_gamma":
            dist = ZeroInflatedGamma(a, torch.exp(self.log_c), gate=b)
        elif self.ppp.NOISE_MODEL == "nb":
            dist = pyro.distributions.GammaPoisson(a, torch.exp(self.log_c))
        elif self.ppp.NOISE_MODEL == "zip":
            dist = pyro.distributions.ZeroInflatedPoisson(a, gate=b)
        elif self.ppp.NOISE_MODEL == "log_normal":
            dist = pyro.distributions.LogNormal(a, torch.exp(self.log_c))
        else:
            raise RuntimeError()
        return dist

    def reconstruction_likelihood(self, a, b, x, corrupted_entries):
        dist = self.get_dist(a, b)
        # measure prob of seeing image under p(x|z)
        # bad variable naming :P
        zero = torch.tensor([2.0]).to(a.device)
        if torch.any(dist.log_prob(zero).isinf()):
            print("infinite value detected")
            raise RuntimeError("manual abort")
        if torch.any(dist.log_prob(zero).isnan()):
            print("nan value detected")
            raise RuntimeError("manual abort")
        if self.ppp.NOISE_MODEL in ["gamma, zi_gamma", "log_normal"]:
            offset = 1e-4
        else:
            offset = 0.0
        log_pxz = dist.log_prob(x + offset)
        if self.mask_loss:
            log_pxz[corrupted_entries] = 0.0
        non_corrupted_count = corrupted_entries.logical_not().sum()
        s = log_pxz.sum(dim=-1) / non_corrupted_count
        return s

    def expected_value(self, a, b=None):
        dist = self.get_dist(a, b)
        return dist.mean

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        if (
            torch.isnan(mu).any()
            or torch.isnan(std).any()
            or torch.isinf(mu).any()
            or torch.isinf(std).any()
        ):
            print("nan or inf in (mu, std) detected (computing the kl)!")
            raise RuntimeError("manual abort")
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def loss_function(self, x, a, b, mu, std, z, corrupted_entries):
        # reconstruction loss
        # print(x_hat.shape)
        cm = get_detect_anomaly_cm(self.ppp)
        with cm:
            if self.ppp.MONTE_CARLO:
                if (
                    torch.isnan(a).any()
                    or torch.isnan(b).any()
                    or torch.isinf(a).any()
                    or torch.isinf(b).any()
                ):
                    print("nan or inf in (a, b) detected!")
                    # this recon_loss will contain a NaN and NaN will be propagated to elbo, which will trigger a
                    # print and optuna_nan_workaround()
                    recon_loss = torch.sum(a, dim=1) + torch.sum(b, dim=1)
                    raise RuntimeError("manual abort")
                else:
                    recon_loss = self.reconstruction_likelihood(
                        a, b, x, corrupted_entries
                    )
                # kl
                kl = self.kl_divergence(z, mu, std)
            else:
                recon_loss = -self.mse_loss(a, x, corrupted_entries)
                log_var = 2 * torch.log(std)
                kl = self.kld_loss(mu, log_var)
            elbo = self.optuna_parameters["vae_beta"] * kl - recon_loss
            elbo = elbo.mean()
            if torch.isinf(elbo).any() or torch.isnan(elbo).any():
                print("nan or inf in loss detected!")
                raise RuntimeError("manual abort")
            elbo = optuna_nan_workaround(elbo)
            return elbo, kl, recon_loss

    def forward(self, x):
        cm = get_detect_anomaly_cm(self.ppp)
        with cm:
            # x_encoded = self.encoder(x)
            # mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
            mu, log_var = self.encoder(x)

            # sample z from q
            eps = 1e-7
            std = torch.exp(log_var / 2) + eps
            if (
                torch.isnan(mu).any()
                or torch.isnan(std).any()
                or torch.isinf(mu).any()
                or torch.isinf(std).any()
            ):
                print("nan or inf in (mu, std) detected!")
                raise RuntimeError("manual abort")
            try:
                q = torch.distributions.Normal(mu, std)
            except ValueError:
                print("ooo")
            z = q.rsample()

            # decoded
            a, b = self.decoder(z)
            if (
                torch.isnan(a).any()
                or torch.isnan(mu).any()
                or torch.isnan(std).any()
                or torch.isnan(z).any()
            ):
                print("nan in forward detected!")
                raise RuntimeError("manual abort")

            # print('so far so good')
            return a, b, mu, std, z

    def training_step(self, batch, batch_idx):
        # print('min, max:', batch.min().cpu().detach(), batch.max().cpu().detach())
        expression, is_corrupted = batch
        assert len(expression.shape) == 2
        # encode x to get the mu and variance parameters
        a, b, mu, std, z = self.forward(expression)
        elbo, kl, recon_loss = self.loss_function(
            expression, a, b, mu, std, z, is_corrupted
        )
        if elbo is None:
            self.log_dict(
                {
                    "elbo": torch.tensor(float("nan")),
                    "kl": torch.tensor(float("nan")),
                    "reconstruction": torch.tensor(float("nan")),
                }
            )
            return torch.tensor(float("nan"))
        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "reconstruction": recon_loss.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx, dataloader_idx):
        expression, is_corrupted = batch
        assert len(expression.shape) == 2
        a, b, mu, std, z = self.forward(expression)
        elbo, kl, recon_loss = self.loss_function(
            expression, a, b, mu, std, z, is_corrupted
        )
        if elbo is None:
            self.logger.log_hyperparams(params={}, metrics={"hp_metric": elbo})
            d = {
                "elbo": torch.tensor(float("nan")),
                "kl": torch.tensor(float("nan")),
                "reconstruction": torch.tensor(float("nan")),
            }
            return d
        else:
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
                    # self.log(f'epoch_{k} {phase}', avg_loss, on_epoch=False)
                    if phase == "validation" and k == "elbo":
                        batch_val_elbo = avg_loss
            assert batch_val_elbo is not None
            self.log("batch_val_elbo", batch_val_elbo)


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
