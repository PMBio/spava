##
import math

import pyro
import pyro.distributions
import pytorch_lightning as pl
import torch
from torch import nn

from analyses.torch_boilerplate import (
    optuna_nan_workaround,
    ZeroInflatedGamma,
    ZeroInflatedNormal,
)

pl.seed_everything(1234)

from utils import get_execute_function
from analyses.torch_boilerplate import get_detect_anomaly_cm

e_ = get_execute_function()


class VAE(pl.LightningModule):
    def __init__(
        self,
        optuna_parameters,
        in_channels,
        out_channels,
        mask_loss: bool = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(kwargs)
        # self.save_hyperparameters()
        self.optuna_parameters = optuna_parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = self.optuna_parameters["vae_latent_dims"]
        self.mask_loss = mask_loss
        self.dropout_alpha = self.optuna_parameters["dropout_alpha"]

        ##
        def get_dims(n):
            dims = []
            max_d = 500
            d = n
            k = 0.75
            for i in range(2):
                d = math.ceil(d * k)
                max_d = math.ceil(max_d * k)
                dims.append(min(d, max_d))
            return dims

        from analyses.torch_boilerplate import get_fc_layers, get_conv_layers

        # encoder stuff
        n = self.in_channels
        dims = [n, 2 * n, 4 * n, 4 * n]
        self.conv_encoder = get_conv_layers(dims=dims, kernel_sizes=[5, 3, 3], name='conv_encoder')
        m = self.conv_encoder(torch.zeros(1, n, 32, 32))
        d = 20
        self.encoder_fc = get_fc_layers(dims=[m.numel(), d], name='encoder_fc', dropout_alpha=self.dropout_alpha)

        self.encoder_mean = nn.Linear(d, self.latent_dim)
        self.encoder_log_var = nn.Linear(d, self.latent_dim)

        # decoder stuff
        dims = get_dims(self.in_channels)
        decoder_dims = [self.latent_dim] + list(reversed(dims))
        print(f"decoder_dims = {decoder_dims}")
        self.decoder_fc = get_fc_layers(
            dims=decoder_dims, name="decoder_fc", dropout_alpha=self.dropout_alpha
        )
        self.decoder_a = nn.Linear(dims[0], self.out_channels)
        self.softplus = nn.Softplus()
        self.decoder_b = nn.Linear(dims[0], self.out_channels)
        self.sigmoid = nn.Sigmoid()

        self.log_c = nn.Parameter(
            torch.Tensor([self.optuna_parameters["log_c"]] * self.out_channels)
        )
        self.logit_d = nn.Parameter(
            torch.logit(torch.Tensor([0.001] * self.in_channels))
        )

    def encoder(self, x):
        x = self.conv_encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_fc(x)
        mu = self.encoder_mean(x)
        log_var = self.encoder_log_var(x)
        return mu, log_var

    def decoder(self, z):
        z = self.decoder_fc(z)
        decoded_a = self.decoder_a(z)
        decoded_b = self.decoder_b(z)
        eps = 1e-4

        if self._hparams["NOISE_MODEL"] in ["gamma", "zi_gamma", "nb"]:
            decoded_a = self.softplus(decoded_a) + eps  # + 2
        elif self._hparams["NOISE_MODEL"] in ["zinb"]:
            decoded_a = self.softplus(decoded_a) + eps  # + 2
        elif self._hparams["NOISE_MODEL"] in ["zip", "log_normal"]:
            decoded_a = self.softplus(decoded_a)
        elif self._hparams["NOISE_MODEL"] in ["gaussian", "zin"]:
            pass
        else:
            assert False

        if self._hparams["NOISE_MODEL"] in ["zip", "zin", "zi_gamma", 'zinb']:
            decoded_b = self.sigmoid(
                decoded_b / (self.optuna_parameters["learning_rate"] * 10)
            )
        elif self._hparams["NOISE_MODEL"] in ["gaussian", "log_normal"]:
            pass
        elif self._hparams["NOISE_MODEL"] in ['gamma', 'nb']:
            raise NotImplementedError()
        else:
            assert False

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
        if self._hparams["NOISE_MODEL"] == "gaussian":
            dist = pyro.distributions.Normal(a, torch.exp(self.log_c))
        elif self._hparams["NOISE_MODEL"] == "zin":
            dist = ZeroInflatedNormal(a, torch.exp(self.log_c), gate=b)
        elif self._hparams["NOISE_MODEL"] == "gamma":
            dist = pyro.distributions.Gamma(a, torch.exp(self.log_c))
        elif self._hparams["NOISE_MODEL"] == "zi_gamma":
            dist = ZeroInflatedGamma(a, torch.exp(self.log_c), gate=b)
        elif self._hparams["NOISE_MODEL"] == "nb":
            dist = pyro.distributions.GammaPoisson(a, torch.exp(self.log_c))
        elif self._hparams["NOISE_MODEL"] == "zip":
            dist = pyro.distributions.ZeroInflatedPoisson(a, gate=b)
        elif self._hparams["NOISE_MODEL"] == "log_normal":
            dist = pyro.distributions.LogNormal(a, torch.exp(self.log_c))
        elif self._hparams["NOISE_MODEL"] == "zinb":
            dist = pyro.distributions.ZeroInflatedNegativeBinomial(
                a, probs=torch.sigmoid(self.log_c), gate=b
            )
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
        if self._hparams["NOISE_MODEL"] in ["gamma, zi_gamma", "log_normal"]:
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
        cm = get_detect_anomaly_cm(self._hparams["DETECT_ANOMALY"])
        with cm:
            if self._hparams["MONTE_CARLO"]:
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
        cm = get_detect_anomaly_cm(self._hparams["DETECT_ANOMALY"])
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

    def unfold_batch(self, batch):
        raster, mask, expression, is_corrupted = batch
        assert len(expression.shape) == 2
        image_input = torch.cat((raster, mask), dim=-1)
        image_input = image_input.permute((0, 3, 1, 2))
        return image_input, expression, is_corrupted

    def training_step(self, batch, batch_idx):
        # print('min, max:', batch.min().cpu().detach(), batch.max().cpu().detach())
        image_input, expression, is_corrupted = self.unfold_batch(batch)

        a, b, mu, std, z = self.forward(image_input)
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
        image_input, expression, is_corrupted = self.unfold_batch(batch)

        a, b, mu, std, z = self.forward(image_input)
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
