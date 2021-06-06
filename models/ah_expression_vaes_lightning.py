class Ppp:
    pass


ppp = Ppp()

# ppp.DEBUG_TORCH = 'yessss'
ppp.MAX_EPOCHS = 20
ppp.BATCH_SIZE = 1024
ppp.LEARNING_RATE = 0.8e-3
ppp.LOG_C = 1
ppp.VAE_BETA = 1e-2
ppp.MONTE_CARLO = True
ppp.MASK_LOSS = True
ppp.LATENT_DIMS = 10
# ppp.DEBUG = True
ppp.DEBUG = False
if ppp.DEBUG:
    ppp.NUM_WORKERS = 0
    ppp.DETECT_ANOMALY = True
else:
    if 'DEBUG_TORCH' in ppp.__dict__:
        ppp.NUM_WORKERS = 0
    else:
        ppp.NUM_WORKERS = 16
    ppp.DETECT_ANOMALY = False
# ppp.NOISE_MODEL = 'gaussian'
# ppp.NOISE_MODEL = 'gamma'
ppp.NOISE_MODEL = 'zip'
# ppp.NOISE_MODEL = 'zin'
# ppp.NOISE_MODEL = 'log_normal'


# ppp.NOISE_MODEL = 'zi_gamma'
# ppp.NOISE_MODEL = 'nb'

def set_ppp_from_loaded_model(pl_module):
    global ppp
    ppp = pl_module.hparams


import os.path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

pl.seed_everything(1234)

from torch import nn

from models.ag_resnet_vae import resnet_encoder, resnet_decoder
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import PIL
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from torch import autograd
import contextlib
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import pyro
import pyro.distributions

import torchvision.transforms
from data2 import CellDataset

quantiles_for_normalization = np.array([4.0549, 1.8684, 1.3117, 3.8141, 2.6172, 3.1571, 1.4984, 1.8866, 1.2621,
                                        3.7035, 3.6496, 1.8566, 2.5784, 0.9939, 1.4314, 2.1803, 1.8672, 1.6674,
                                        2.3555, 0.8917, 5.1779, 1.8002, 1.4042, 2.3873, 1.0509, 1.0892, 2.2708,
                                        3.4417, 1.8348, 1.8449, 2.8699, 2.2071, 1.0464, 2.5855, 2.0384, 4.8609,
                                        2.0277, 3.3281, 3.9273])


class ImageSampler(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        trainer.logger.experiment.add_scalars('c', {f'channel{i}': torch.exp(pl_module.log_c[i]) for i in range(len(
            pl_module.log_c))}, trainer.global_step)
        trainer.logger.experiment.add_scalars('d', {f'channel{i}': torch.sigmoid(pl_module.logit_d[i]) for i in range(
            len(
                pl_module.logit_d))}, trainer.global_step)
        # for dataloader_idx in [0, 1]:
        # loader = trainer.val_dataloaders[dataloader_idx]
        # dataloader_label = 'training' if dataloader_idx == 0 else 'validation'
        # trainer.logger.experiment.add_image(f'reconstruction/{dataloader_label}', img,
        #                                     trainer.global_step)

        # trainer.logger.experiment.add_histogram(f'histograms/{dataloader_label}/image{i}/channel'
        #                                         f'{c}/original', original_masked_c[0].flatten(),
        #                                         trainer.global_step)


def get_detect_anomaly_cm():
    if ppp.DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


from pyro.distributions.zero_inflated import ZeroInflatedDistribution
from pyro.distributions import Gamma
from torch.distributions import constraints


class ZeroInflatedNormal(ZeroInflatedDistribution):
    """
    A Zero Inflated Normal distribution.
    """
    arg_constraints = {"loc": constraints.real,
                       "scale": constraints.positive,
                       "gate": constraints.unit_interval,
                       "gate_logits": constraints.real}
    support = constraints.real

    def __init__(self, loc, scale, *, gate=None, gate_logits=None,
                 validate_args=None):
        base_dist = pyro.distributions.Normal(
            loc=loc,
            scale=scale,
            validate_args=False,
        )
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

    @property
    def location(self):
        return self.base_dist.mean

    @property
    def stddev(self):
        return self.base_dist.stddev


class ZeroInflatedGamma(ZeroInflatedDistribution):
    """
    A Zero Inflated Gamma distribution.

    :param concentration: shape parameter of the distribution (often referred to as alpha).
    :type concentration: float or torch.Tensor
    :param rate: rate = 1 / scale of the distribution (often referred to as beta).
    :type float or torch.Tensor
    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor gate_logits: logits of extra zeros.
    """
    arg_constraints = {"concentration": constraints.greater_than(0.),
                       "rate": constraints.greater_than(0.),
                       "gate": constraints.unit_interval,
                       "gate_logits": constraints.real}
    support = constraints.greater_than(0.)

    def __init__(self, concentration, rate, *, gate=None, gate_logits=None,
                 validate_args=None):
        base_dist = Gamma(
            concentration=concentration,
            rate=rate,
            validate_args=False,
        )
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

    @property
    def concentration(self):
        return self.base_dist.concentration

    @property
    def rate(self):
        return self.base_dist.rate


# dist = Gamma(10, 10)
# dist = ZeroInflatedGamma(10, 10, gate=torch.tensor([0.1]))
# x = dist.sample((10000,))
# plt.figure()
# plt.hist(x.numpy(), bins=100)
# plt.show()

class VAE(pl.LightningModule):
    def __init__(self, in_channels, latent_dim=10, out_channels=None, mask_loss: bool = None, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters()
        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = self.in_channels
        else:
            self.out_channels = out_channels
        self.latent_dim = latent_dim
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

        self.log_c = nn.Parameter(torch.Tensor([ppp.LOG_C] * 39))
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
        if ppp.NOISE_MODEL in ['gamma', 'zi_gamma', 'nb']:
            decoded_a = self.softplus(decoded_a) + 2
        elif ppp.NOISE_MODEL == 'zip':
            decoded_a = self.softplus(decoded_a)
        if ppp.NOISE_MODEL in ['zip', 'zig', 'zi_gamma']:
            decoded_b = self.sigmoid(decoded_b)
        return decoded_a, decoded_b

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=ppp.LEARNING_RATE)

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
            se_loss[corrupted_entries] = 0.
        non_corrupted_count = corrupted_entries.shape[-1] - corrupted_entries.sum(dim=-1)
        mse_loss = se_loss.sum(dim=-1) / non_corrupted_count
        return mse_loss

    def get_dist(self, a, b):
        if ppp.NOISE_MODEL == 'gaussian':
            dist = pyro.distributions.Normal(a, torch.exp(self.log_c))
        elif ppp.NOISE_MODEL == 'zin':
            dist = ZeroInflatedNormal(a, torch.exp(self.log_c), gate=b)
        elif ppp.NOISE_MODEL == 'gamma':
            dist = pyro.distributions.Gamma(a, torch.exp(self.log_c))
        elif ppp.NOISE_MODEL == 'zi_gamma':
            dist = ZeroInflatedGamma(a, torch.exp(self.log_c), gate=b)
        elif ppp.NOISE_MODEL == 'nb':
            dist = pyro.distributions.GammaPoisson(a, torch.exp(self.log_c))
        elif ppp.NOISE_MODEL == 'zip':
            dist = pyro.distributions.ZeroInflatedPoisson(a, gate=b)
        elif ppp.NOISE_MODEL == 'log_normal':
            dist = pyro.distributions.LogNormal(a, torch.exp(self.log_c))
        else:
            raise RuntimeError()
        return dist

    def reconstruction_likelihood(self, a, b, x, corrupted_entries):
        dist = self.get_dist(a, b)
        # measure prob of seeing image under p(x|z)
        # bad variable naming :P
        zero = torch.tensor([2.]).to(a.device)
        if torch.any(dist.log_prob(zero).isinf()):
            print('infinite value detected')
        if torch.any(dist.log_prob(zero).isnan()):
            print('nan value detected')
        if ppp.NOISE_MODEL in ['gamma, zi_gamma', 'log_normal']:
            offset = 1e-4
        else:
            offset = 0.
        log_pxz = dist.log_prob(x + offset)
        if self.mask_loss:
            log_pxz[corrupted_entries] = 0.
        non_corrupted_count = corrupted_entries.shape[-1] - corrupted_entries.sum(dim=-1)
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
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def loss_function(self, x, a, b, mu, std, z, corrupted_entries):
        # reconstruction loss
        # print(x_hat.shape)
        cm = get_detect_anomaly_cm()
        with cm:
            if ppp.MONTE_CARLO:
                recon_loss = self.reconstruction_likelihood(a, b, x, corrupted_entries)
                # kl
                kl = self.kl_divergence(z, mu, std)
                # elbo
                # elbo = kl - recon_loss
            else:
                recon_loss = - self.mse_loss(a, x, corrupted_entries)
                log_var = 2 * torch.log(std)
                kl = self.kld_loss(mu, log_var)
            elbo = (ppp.VAE_BETA * kl - recon_loss)
            elbo = elbo.mean()
            if torch.isnan(elbo).any():
                print('nan in loss detected!')
            if torch.isinf(elbo).any():
                print('inf in loss detected!')
            return elbo, kl, recon_loss

    def forward(self, x):
        cm = get_detect_anomaly_cm()
        with cm:
            # x_encoded = self.encoder(x)
            # mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
            mu, log_var = self.encoder(x)

            # sample z from q
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # decoded
            a, b = self.decoder(z)
            if torch.isnan(a).any() or torch.isnan(mu).any() or torch.isnan(std).any() or torch.isnan(z).any():
                print('nan in forward detected!')
            # print('so far so good')
            return a, b, mu, std, z

    def training_step(self, batch, batch_idx):
        # print('min, max:', batch.min().cpu().detach(), batch.max().cpu().detach())
        x, corrupted_entries = batch
        # assert x.shape == (BATCH_SIZE, 39)
        assert len(x.shape) == 2
        assert x.shape[-1] == 39
        # encode x to get the mu and variance parameters
        a, b, mu, std, z = self.forward(x)
        elbo, kl, recon_loss = self.loss_function(x, a, b, mu, std, z, corrupted_entries)

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'reconstruction': recon_loss.mean(),
        })

        return elbo

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, corrupted_entries = batch
        # assert x.shape == (BATCH_SIZE, 39)
        assert len(x.shape) == 2
        assert x.shape[-1] == 39
        a, b, mu, std, z = self.forward(x)
        elbo, kl, recon_loss = self.loss_function(x, a, b, mu, std, z, corrupted_entries)

        d = {
            'elbo': elbo,
            'kl': kl.mean(),
            'reconstruction': recon_loss.mean()
        }
        return d

    def validation_epoch_end(self, outputs):
        if not self.trainer.running_sanity_check:
            assert type(outputs) is list
            for i, o in enumerate(outputs):
                for k in ['elbo', 'kl', 'reconstruction']:
                    avg_loss = torch.stack([x[k] for x in o]).mean().cpu().detach()
                    phase = 'training' if i == 0 else 'validation'
                    self.logger.experiment.add_scalar(f'avg_metric/{k}/{phase}', avg_loss, self.global_step)
                    # self.log(f'epoch_{k} {phase}', avg_loss, on_epoch=False)
                # return {'log': d}


# from https://medium.com/@adrian.waelchli/3-simple-tricks-that-will-change-the-way-you-debug-pytorch-5c940aa68b03
class LogComputationalGraph(pl.Callback):
    def __init__(self):
        self.already_logged = False

    def on_validation_start(self, trainer: pl.Trainer, pl_module):
        if not trainer.running_sanity_check:
            if not self.already_logged:
                return
                # this code causes a TracerWarning
                # self.already_logged = True
                # sample_image = torch.rand((BATCH_SIZE, len(COOL_CHANNELS), 32, 32))
                # pl_module.logger.experiment.add_graph(VAE(), sample_image)


# class LogHyperparameters(pl.Callback):
#     def __init__(self):
#         self.alredy_logged = False
#
#     def on_validation_start(self, trainer, pl_module) -> None:
#         if not trainer.running_sanity_check:
#             if not self.alredy_logged:
#                 pl_module.logger.log_hyperparams(ppp.__dict__)
#                 self.alredy_logged = True

from data2 import AccumulatedDataset, FilteredMasksRelabeled, ExpressionDataset, file_path


class PerturbedCellDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.ds = AccumulatedDataset(split, feature='mean', from_raw=True, transform=False)
        self.index_converter = FilteredMasksRelabeled(split).get_indices_conversion_arrays
        f = file_path('ah_filtered_untransformed_expression_tensor_merged.npy')
        # os.remove(f)
        if not os.path.isfile(f):
            all = []
            for i in tqdm(range(len(self.ds)), desc='merging expression tensor'):
                e = self.ds[i]
                new_e = ExpressionDataset.expression_old_to_new(e, i, index_converter=self.index_converter)
                all.append(new_e)
            merged = np.concatenate(all, axis=0)
            np.save(f, merged)
        self.merged = torch.tensor(np.load(f))
        self.merged = torch.asinh(self.merged)
        self.merged /= quantiles_for_normalization
        self.merged = self.merged.float()
        self.corrupted_entries = torch.zeros_like(self.merged, dtype=torch.bool)
        self.original_merged = None
        self.seed = None

    def perturb(self, seed=0):
        self.seed = seed
        from torch.distributions import Bernoulli
        dist = Bernoulli(probs=0.1)
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.corrupted_entries = dist.sample(self.merged.shape).bool()
        torch.set_rng_state(state)
        self.original_merged = self.merged.clone()
        self.merged[self.corrupted_entries] = 0.

    def __len__(self):
        return len(self.merged)

    def __getitem__(self, i):
        return self.merged[i, :], self.corrupted_entries[i, :]


def train(perturb=False):
    # parser = ArgumentParser()
    # parser.add_argument('--gpus', type=int, default=1)
    # args = parser.parse_args()

    train_ds = PerturbedCellDataset('train')
    if perturb:
        train_ds.perturb()
    train_ds_validation = PerturbedCellDataset('train')
    val_ds = PerturbedCellDataset('validation')
    print(f'len(train_ds) = {len(train_ds[0])}, train_ds[0] = {train_ds[0][0]}, train_ds[0].shape ='
          f' {train_ds[0][0].shape}')

    from data2 import file_path
    logger = TensorBoardLogger(save_dir=file_path('checkpoints'), name='expression_vae')
    print(f'logging in {logger.experiment.log_dir}')
    checkpoint_callback = ModelCheckpoint(dirpath=file_path(f'{logger.experiment.log_dir}/checkpoints'),
                                          monitor='elbo',
                                          # every_n_train_steps=2,
                                          save_last=True,
                                          save_top_k=3)
    trainer = pl.Trainer(gpus=1, max_epochs=ppp.MAX_EPOCHS,
                         callbacks=[ImageSampler(), LogComputationalGraph(), checkpoint_callback],
                         logger=logger, num_sanity_val_steps=0,  # track_grad_norm=2,
                         log_every_n_steps=15 if not ppp.DEBUG else 1, val_check_interval=1 if ppp.DEBUG else 400)
    # set back val_check_interval to 200

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
    train_loader = DataLoader(d, batch_size=ppp.BATCH_SIZE, num_workers=ppp.NUM_WORKERS, pin_memory=True,
                              shuffle=True)
    train_loader_batch = DataLoader(train_subset, batch_size=ppp.BATCH_SIZE, num_workers=ppp.NUM_WORKERS,
                                    pin_memory=True)

    indices = np.random.choice(len(val_ds), n, replace=False)
    val_subset = Subset(val_ds, indices)
    val_loader = DataLoader(val_subset, batch_size=ppp.BATCH_SIZE, num_workers=ppp.NUM_WORKERS, pin_memory=True)
    #
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
    vae = VAE(in_channels=39, latent_dim=ppp.LATENT_DIMS, out_channels=None, mask_loss=ppp.MASK_LOSS, **ppp.__dict__)
    trainer.fit(vae, train_dataloader=train_loader, val_dataloaders=[train_loader_batch, val_loader])
    print(f'finished logging in {logger.experiment.log_dir}')


if __name__ == '__main__':
    train(perturb=True)
