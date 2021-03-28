import math

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(1234)

import itertools
from torch import nn
import torch

from models.ae_resnet_vae import resnet_encoder, resnet_decoder
# from pl_bolts.models.autoencoders.components import (
# resnet18_decoder,
# resnet18_encoder,
# )
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('Agg')
import functools
import numpy as np
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

# from pl_bolts.utils import _TORCHVISION_AVAILABLE
# from pl_bolts.utils.warnings import warn_missing_pkg

from torchvision import transforms
from data2 import CellDataset

# def imagenet_normalization():
#     if not _TORCHVISION_AVAILABLE:  # pragma: no cover
#         raise ModuleNotFoundError(
#             'You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`.'
#         )
#
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     return normalize
#
#
COOL_CHANNELS = np.array([0, 10, 20])
BATCH_SIZE = 1024
LEARNING_RATE = 0.8e-3
# DEBUG = True
DEBUG = False
if DEBUG:
    NUM_WORKERS = 0
    DETECT_ANOMALY = True
else:
    # NUM_WORKERS = 0
    NUM_WORKERS = 16
    DETECT_ANOMALY = False

# def ome_normalization():
#     mean = np.array([0.3128328, 0.08154685, 0.02617499, 0.31149776, 0.10011313,
#                      0.13143819, 0.04897958, 0.05522078, 0.02628855, 0.12524123,
#                      0.15552816, 0.08004793, 0.13349437, 0.02045013, 0.04155505,
#                      0.07637688, 0.05526352, 0.04818857, 0.11221485, 0.01779799,
#                      0.53215206, 0.08219107, 0.03510931, 0.08550659, 0.02237169,
#                      0.02657647, 0.09854327, 0.22031476, 0.04274541, 0.06778383,
#                      0.24079644, 0.09004467, 0.0234197, 0.13312621, 0.04914724,
#                      0.29719813, 0.10172928, 0.18843424, 0.25893724])
#     std = np.sqrt(np.array([0.81152901, 0.11195328, 0.03844969, 0.76020458, 0.19636732,
#                             0.30648388, 0.06448294, 0.08879372, 0.03747649, 0.32956727,
#                             0.40133228, 0.11878445, 0.24177647, 0.02510913, 0.05398327,
#                             0.15110854, 0.09525968, 0.07278724, 0.17972434, 0.01950939,
#                             1.73329118, 0.11334923, 0.04934192, 0.15689578, 0.02762272,
#                             0.03045641, 0.16039316, 0.49438282, 0.07485281, 0.10151964,
#                             0.45035213, 0.15424273, 0.02854364, 0.23177609, 0.09494518,
#                             0.98995058, 0.14861627, 0.41785507, 0.66190155]))
#     mean = mean[COOL_CHANNELS]
#     std = std[COOL_CHANNELS]
#     normalize = transforms.Normalize(mean=mean, std=std)
#     return normalize

quantiles_for_normalization = np.array([4.0549, 1.8684, 1.3117, 3.8141, 2.6172, 3.1571, 1.4984, 1.8866, 1.2621,
                                        3.7035, 3.6496, 1.8566, 2.5784, 0.9939, 1.4314, 2.1803, 1.8672, 1.6674,
                                        2.3555, 0.8917, 5.1779, 1.8002, 1.4042, 2.3873, 1.0509, 1.0892, 2.2708,
                                        3.4417, 1.8348, 1.8449, 2.8699, 2.2071, 1.0464, 2.5855, 2.0384, 4.8609,
                                        2.0277, 3.3281, 3.9273])[COOL_CHANNELS]
# print('fino a qui tutto bene')

def get_image(loader, model):
    all_originals = []
    all_originals_masked = []
    all_reconstructed = []
    all_reconstructed_masked = []
    mask_color = torch.tensor([x / 255 for x in [254, 112, 31]]).float()
    new_size = (128, 128)
    upscale = transforms.Resize(new_size, interpolation=PIL.Image.NEAREST)
    n = 15
    with torch.no_grad():
        batch = loader.__iter__().__next__()
        omes = batch[0]
        masks = batch[1]
        assert len(omes.shape) == 4
        assert len(omes) >= n
        data = omes[:n].to(model.device)
        masks_data = masks[:n].to(model.device)
        pred = model.forward(data, masks_data)[0]
    n_channels = data.shape[1]
    all_original_c = {c: [] for c in range(n_channels)}
    all_original_masked_c = {c: [] for c in range(n_channels)}
    all_reconstructed_c = {c: [] for c in range(n_channels)}
    all_reconstructed_masked_c = {c: [] for c in range(n_channels)}

    for i in range(n):
        original = data[i].cpu().permute(1, 2, 0) * quantiles_for_normalization
        r_hat = pred[i].cpu().permute(1, 2, 0)
        p = torch.sigmoid(model.negative_binomial_p_logit).cpu().detach()
        mean = model.negative_binomial_mean(r=r_hat, p=p)
        reconstructed = mean * quantiles_for_normalization

        a_original = original.amin(dim=(0, 1))
        b_original = original.amax(dim=(0, 1))
        m = masks_data[i].cpu().bool()
        mm = torch.squeeze(m, 0)
        reconstructed_flattened = torch.reshape(reconstructed, (-1, reconstructed.shape[-1]))
        mask_flattened = mm.flatten()
        a_reconstructed = reconstructed_flattened[mask_flattened, :].amin(dim=0)
        b_reconstructed = reconstructed_flattened[mask_flattened, :].amax(dim=0)
        a = torch.min(a_original, a_reconstructed)
        b = torch.max(b_original, b_reconstructed)

        original = ((original - a) / (b - a)).float()
        reconstructed = ((reconstructed - a) / (b - a)).float()

        mm_not = torch.logical_not(mm)
        assert torch.all(reconstructed[mm, :] >= 0.)
        assert torch.all(reconstructed[mm, :] <= 1.)
        reconstructed = torch.clamp(reconstructed, 0., 1.)

        original_masked = original.clone()
        original_masked[mm_not, :] = mask_color
        reconstructed_masked = reconstructed.clone()
        reconstructed_masked[mm_not, :] = mask_color

        for c in range(n_channels):
            original_c = original[:, :, c]
            original_c = torch.stack([original_c] * 3, dim=2)

            reconstructed_c = reconstructed[:, :, c]
            reconstructed_c = torch.stack([reconstructed_c] * 3, dim=2)

            def f(t):
                t = t.permute(2, 0, 1)
                t = upscale(t)
                return t

            def overlay_mask(t):
                t = t.clone()
                t[mm_not, :] = mask_color
                return t

            a_original_c = original_c.amin(dim=(0, 1))
            b_original_c = original_c.amax(dim=(0, 1))
            reconstructed_flattened_c = torch.reshape(reconstructed_c, (-1, reconstructed_c.shape[-1]))
            mask_flattened = mm.flatten()
            a_reconstructed_c = reconstructed_flattened_c[mask_flattened, :].amin(dim=0)
            b_reconstructed_c = reconstructed_flattened_c[mask_flattened, :].amax(dim=0)
            a_c = torch.min(a_original_c, a_reconstructed_c)
            b_c = torch.max(b_original_c, b_reconstructed_c)

            t = (original_c - a_c) / (b_c - a_c)
            all_original_c[c].append(f(t))
            all_original_masked_c[c].append(f(overlay_mask(t)))
            t = (reconstructed_c - a_c) / (b_c - a_c)
            all_reconstructed_c[c].append(f(t))
            all_reconstructed_masked_c[c].append(f(overlay_mask(t)))

        original = upscale(original.permute(2, 0, 1))
        reconstructed = upscale(reconstructed.permute(2, 0, 1))
        original_masked = upscale(original_masked.permute(2, 0, 1))
        reconstructed_masked = upscale(reconstructed_masked.permute(2, 0, 1))

        all_originals.append(original)
        all_reconstructed.append(reconstructed)
        all_originals_masked.append(original_masked)
        all_reconstructed_masked.append(reconstructed_masked)

    l = all_originals + all_reconstructed + all_originals_masked + all_reconstructed_masked
    for c in range(n_channels):
        l += (all_original_c[c] + all_reconstructed_c[c] + all_original_masked_c[c] + all_reconstructed_masked_c[c])

    img = make_grid(l, nrow=n)
    return img


# plt.figure(figsize=(30, 30))
# im = img.permute(1, 2, 0).numpy()
# print(im.shape, im.min(), im.max())
# plt.imshow(im)
# plt.show()

class ImageSampler(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        # def on_train_epoch_end(self, trainer: pl.Trainer, pl_module, outputs):
        # Z COMES FROM NORMAL(0, 1)
        # rand_v = torch.rand((self.num_preds, pl_module.hparams.latent_dim), device=pl_module.device)
        # p = torch.distributions.Normal(torch.zeros_like(rand_v), torch.ones_like(rand_v))
        # z = p.rsample()

        # normalize = ome_normalization()
        a = pl_module.negative_binomial_p_logit
        b = pl_module.boosted_sigmoid(a)
        trainer.logger.experiment.add_scalars(f'negative_binomial_p_logit', {f'channel{i}': a[i] for i in range(len(
            a))}, trainer.global_step)
        trainer.logger.experiment.add_scalars(f'negative_binomial_p', {f'channel{i}': b[i] for i in range(len(b))},
                                              trainer.global_step)

        for dataloader_idx in [0, 1]:
            loader = trainer.val_dataloaders[dataloader_idx]
            dataloader_label = 'training' if dataloader_idx == 0 else 'validation'
            img = get_image(loader, pl_module)
            trainer.logger.experiment.add_image(f'reconstruction/{dataloader_label}', img,
                                                trainer.global_step)

            # trainer.logger.experiment.add_histogram(f'histograms/{dataloader_label}/image{i}/channel'
            #                                         f'{c}/original', original_masked_c[0].flatten(),
            #                                         trainer.global_step)
            # trainer.logger.experiment.add_histogram(f'histograms/{dataloader_label}/image{i}/channel'
            #                                         f'{c}/reconstructed', reconstructed_masked_c[0].flatten(),
            #                                         trainer.global_step)


def get_detect_anomaly_cm():
    global DETECT_ANOMALY
    if DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


class VAE(pl.LightningModule):
    def __init__(self, n_channels, enc_out_dim=256, latent_dim=64, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.n_channels = n_channels
        self.encoder = resnet_encoder(first_conv=False, maxpool1=False)
        self.decoder = resnet_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)
        self.softplus = nn.Softplus()

        # for the gaussian likelihood
        # self.log_scale = nn.Parameter(torch.Tensor([0.4]))

        # value such that if we apply the sigmoid function we get the p parameter of a negative binomial,
        # one per channel

        VAE.p_booster = 1.
        self.negative_binomial_p_logit = nn.Parameter(
            torch.Tensor([self.boosted_logit(torch.tensor(0.2)).item()] * self.n_channels)
        )
        self.vae_beta = 100.

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    # these functions assume r > 0 and p in ]0, 1[
    @staticmethod
    def negative_binomial_log_prob(r, p, k):
        r = r.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        log_prob = torch.lgamma(k + r) - torch.lgamma(k + 1) - torch.lgamma(r) + r * torch.log(1 - p) + k * torch.log(p)
        return log_prob.permute(0, 3, 1, 2)

    @classmethod
    def boosted_logit(cls, p):
        return torch.logit(p) / cls.p_booster

    @classmethod
    def boosted_sigmoid(cls, t):
        return torch.sigmoid(t * cls.p_booster)

    @staticmethod
    def negative_binomial_mean(r, p):
        return p * r / (1 - p)

    @staticmethod
    def negative_binomial_variance(r, p):
        return p * r / torch.square(1 - p)

    def reconstruction_likelihood(self, x_hat, x, mask):
        # scale = torch.exp(logscale)
        # mean = x_hat
        r = x_hat
        p = self.boosted_sigmoid(self.negative_binomial_p_logit)
        # variance = torch.square(scale)
        # alpha = torch.square(mean) / variance
        # beta = mean / variance
        # dist = torch.distributions.Gamma(alpha, beta)
        log_pxz = self.negative_binomial_log_prob(r, p, k=x)
        # dist = torch.distributions.NegativeBinomial(mean, torch.sigmoid(scale))
        # dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        # log_pxz = dist.log_prob(x)
        if mask is None:
            mask = torch.ones_like(log_pxz)
        s = (mask * log_pxz).mean(dim=(1, 2, 3))
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
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def loss_function(self, x, x_hat, mu, std, z, mask):
        # reconstruction loss
        # print(x_hat.shape)
        cm = get_detect_anomaly_cm()
        with cm:
            recon_loss = self.reconstruction_likelihood(x_hat, x, mask)
            # kl
            kl = self.kl_divergence(z, mu, std)
            # elbo
            elbo = (self.vae_beta * kl - recon_loss)
            elbo = elbo.mean()
            if torch.isnan(elbo).any():
                print('nan in loss detected!')
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
            x_hat = self.decoder(z, mask)
            x_hat = self.softplus(x_hat)
            return x_hat, mu, std, z

    def training_step(self, batch, batch_idx):
        # print('min, max:', batch.min().cpu().detach(), batch.max().cpu().detach())
        x = batch[0]
        mask = batch[1]
        # encode x to get the mu and variance parameters
        x_hat, mu, std, z = self.forward(x, mask)
        elbo, kl, recon_loss = self.loss_function(x, x_hat, mu, std, z, mask)

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'reconstruction': recon_loss.mean(),
        })

        return elbo

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x = batch[0]
        mask = batch[1]
        x_hat, mu, std, z = self.forward(x, mask)
        elbo, kl, recon_loss = self.loss_function(x, x_hat, mu, std, z, mask)

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


class PadByOne:
    def __call__(self, image):
        return F.pad(image, pad=[0, 1, 0, 1], mode='constant', value=0)


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class RGBCells(Dataset):
    def __init__(self, split, augment=False, aggressive_rotation=False):
        assert not (augment is False and aggressive_rotation is True)
        d = {'expression': False, 'center': False, 'ome': True, 'mask': True}
        self.ds = CellDataset(split, d)
        self.augment = augment
        self.aggressive_rotation = aggressive_rotation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            PadByOne()
        ])
        self.augment_transform = transforms.Compose([
            MyRotationTransform(angles=[90, 180, 270]) if not self.aggressive_rotation else transforms.RandomApply(
                nn.ModuleList([transforms.RandomRotation(degrees=360)]),
                p=0.6
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        # self.normalize = ome_normalization()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        x = self.ds[item][0]
        if len(self.ds[item]) == 2:
            mask = self.ds[item][1]
            mask = self.transform(mask)
            if self.augment:
                state = torch.get_rng_state()
                mask = self.augment_transform(mask)
            mask = mask.float()
        elif len(self.ds[item]) == 1:
            mask = None
        else:
            raise ValueError()
        x = self.transform(x)
        if self.augment:
            torch.set_rng_state(state)
            x = self.augment_transform(x)
        x = torch.asinh(x)
        x = x[COOL_CHANNELS, :, :]
        x = x.permute(1, 2, 0)
        # x = (x - self.normalize.mean) / self.normalize.std
        x = x / quantiles_for_normalization
        x = x.permute(2, 0, 1)
        x = x.float()
        return x, mask


def train():
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    train_ds = RGBCells('train', augment=True, aggressive_rotation=True)
    train_ds_validation = RGBCells('train')
    val_ds = RGBCells('validation')

    logger = TensorBoardLogger(save_dir='/data/l989o/spatial_uzh/data/spatial_uzh_processed/a/checkpoints',
                               name='resnet_vae')
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=20, callbacks=[ImageSampler(), LogComputationalGraph()],
                         logger=logger,
                         log_every_n_steps=15, val_check_interval=2 if DEBUG else 50)
    # set back val_check_interval to 200

    if DEBUG:
        n = BATCH_SIZE * 2
    else:
        n = BATCH_SIZE * 20
    indices = np.random.choice(len(train_ds), n, replace=False)
    train_subset = Subset(train_ds_validation, indices)

    if DEBUG:
        d = train_subset
    else:
        d = train_ds
    train_loader = DataLoader(d, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
                              shuffle=True)
    train_loader_batch = DataLoader(train_subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    indices = np.random.choice(len(val_ds), n, replace=False)
    val_subset = Subset(val_ds, indices)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
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
    vae = VAE(n_channels=len(COOL_CHANNELS))
    trainer.fit(vae, train_dataloader=train_loader, val_dataloaders=[train_loader_batch, val_loader])


if __name__ == '__main__':
    train()