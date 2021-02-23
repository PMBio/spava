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
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
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
# import torchvision.transforms.functional as F_vision

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

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
# DEBUG = True
DEBUG = False
if DEBUG:
    NUM_WORKERS = 0
    DETECT_ANOMALY = True
else:
    NUM_WORKERS = 16
    DETECT_ANOMALY = False


def ome_normalization():
    mean = np.array([1.29137214, 0.12204645, 0.04746121, 1.17126412, 0.24452078,
                     0.427998, 0.07071735, 0.11199224, 0.04566163, 0.63347302,
                     0.62917786, 0.13072797, 0.27373635, 0.02843522, 0.06192851,
                     0.38788928, 0.11424681, 0.07840189, 0.2078604, 0.02232897,
                     4.8215692, 0.13145834, 0.05435668, 0.17872389, 0.0315007,
                     0.03429091, 0.20750708, 0.6714512, 0.09881951, 0.12434302,
                     0.51898777, 0.18728622, 0.03190125, 0.28144336, 0.11512508,
                     2.50877083, 0.16205379, 0.52616125, 0.99683675])
    std = np.sqrt(np.array([3.17532575e+01, 3.66818966e-01, 6.65207711e-01, 3.33794102e+01,
                            4.22485386e+00, 9.36283163e+00, 2.23369604e-01, 7.86815906e+00,
                            5.32690521e-01, 4.84695307e+01, 2.08140218e+01, 7.09183370e+00,
                            1.92951659e+00, 9.45509177e-01, 9.75673669e-01, 2.01367976e+02,
                            1.37284794e+00, 3.73569237e-01, 4.87192135e+00, 5.64851603e-01,
                            5.69273662e+02, 6.52422796e+01, 3.68515530e-01, 1.66068873e+00,
                            1.29575157e-01, 6.50012842e-01, 2.25449424e+01, 1.09436277e+01,
                            2.24749223e+00, 8.06681989e+00, 5.34230461e+00, 8.48350188e+00,
                            9.04868194e-02, 3.58260224e+00, 1.58120290e+00, 2.32770610e+02,
                            6.65773423e-01, 6.49080885e+00, 2.20182966e+01]))
    mean = mean[COOL_CHANNELS]
    std = std[COOL_CHANNELS]
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize


class ImageSampler(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module, outputs):
        # Z COMES FROM NORMAL(0, 1)
        # rand_v = torch.rand((self.num_preds, pl_module.hparams.latent_dim), device=pl_module.device)
        # p = torch.distributions.Normal(torch.zeros_like(rand_v), torch.ones_like(rand_v))
        # z = p.rsample()

        normalize = ome_normalization()
        loader = trainer.val_dataloaders[-1]
        all_originals = []
        all_reconstructed = []
        n = 5
        with torch.no_grad():
            batch = loader.__iter__().__next__()
            assert len(batch.shape) == 4
            assert len(batch) >= n
            data = batch[:n]
            pred = pl_module.forward(data.to(pl_module.device))[0]
        for i in range(n):
            def back_to_original(x):
                x = x.permute(1, 2, 0)
                x = x * normalize.std + normalize.mean
                x = x.permute(2, 0, 1)
                return x

            original = back_to_original(data[i].cpu())
            reconstructed = back_to_original(pred[i].cpu())
            a_original = torch.amin(original, dim=(1, 2))
            b_original = torch.amax(original, dim=(1, 2))
            a_reconstructed = reconstructed.amin(dim=(1, 2))
            b_reconstructed = reconstructed.amax(dim=(1, 2))
            a = torch.min(a_original, a_reconstructed)
            b = torch.max(b_original, b_reconstructed)
            f = lambda t, a, b: ((t.permute(1, 2, 0) - a) / (b - a)).permute(2, 0, 1)
            original = f(original, a, b)
            reconstructed = f(reconstructed, a, b)

            new_size = (128, 128)
            compose = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(new_size, interpolation=PIL.Image.NEAREST),
                transforms.ToTensor()
            ])
            original = compose(original)
            reconstructed = compose(reconstructed)
            all_originals.append(original)
            all_reconstructed.append(reconstructed)
        l = all_originals + all_reconstructed
        img = make_grid(l, nrow=n)
        trainer.logger.experiment.add_image('img', img, trainer.current_epoch)


def get_detect_anomaly_cm():
    global DETECT_ANOMALY
    if DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=64, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
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

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.beta = 10.

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

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

    def forward(self, x):
        cm = get_detect_anomaly_cm()
        with cm:
            x_encoded = self.encoder(x)
            mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

            # sample z from q
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # decoded
            x_hat = self.decoder(z)
            return x_hat, mu, std, z

    def loss_function(self, x, x_hat, mu, std, z):
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        # kl
        kl = self.kl_divergence(z, mu, std)
        # elbo
        elbo = (self.beta * kl - recon_loss)
        elbo = elbo.mean()
        return elbo, kl, recon_loss

    def training_step(self, batch, batch_idx):
        # print('min, max:', batch.min().cpu().detach(), batch.max().cpu().detach())
        x = batch
        # encode x to get the mu and variance parameters
        x_hat, mu, std, z = self.forward(x)
        elbo, kl, recon_loss = self.loss_function(x, x_hat, mu, std, z)

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'reconstruction': recon_loss.mean(),
        })
        if torch.isnan(elbo).any():
            print('nan in loss detected!')

        return elbo

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x = batch
        x_hat, mu, std, z = self.forward(x)
        elbo, kl, recon_loss = self.loss_function(x, x_hat, mu, std, z)

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
                    self.logger.experiment.add_scalar(f'avg_metric/{k}/{phase}', avg_loss, self.current_epoch)
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
                self.already_logged = True
                sample_image = torch.rand((BATCH_SIZE, len(COOL_CHANNELS), 32, 32))
                pl_module.logger.experiment.add_graph(VAE(), sample_image)


def train():
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    class PadByOne:
        def __call__(self, image):
            return F.pad(image, pad=[0, 1, 0, 1], mode='constant', value=0)

    class RGBCells(Dataset):
        def __init__(self, split):
            d = {'expression': False, 'center': False, 'ome': True, 'mask': False}
            self.ds = CellDataset(split, d)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                PadByOne()
            ])
            self.normalize = ome_normalization()

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, item):
            x = self.ds[item][0]
            x = self.transform(x)
            x = x[COOL_CHANNELS, :, :]
            x = x.permute(1, 2, 0)
            x = (x - self.normalize.mean) / self.normalize.std
            x = x.permute(2, 0, 1)
            x = torch.asinh(x)
            x = x.float()
            return x

    train_ds = RGBCells('train')
    val_ds = RGBCells('validation')

    logger = TensorBoardLogger(save_dir='/data/l989o/spatial_uzh/data/spatial_uzh_processed/a/checkpoints',
                               name='resnet_vae')
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=20, callbacks=[ImageSampler(), LogComputationalGraph()],
                         logger=logger, log_every_n_steps=5, val_check_interval=2)

    n = BATCH_SIZE * 2
    indices = np.random.choice(len(train_ds), n, replace=False)
    train_subset = Subset(train_ds, indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
                              shuffle=True)
    train_loader_batch = DataLoader(train_subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    n = BATCH_SIZE * 2
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
    vae = VAE()
    trainer.fit(vae, train_dataloader=train_loader, val_dataloaders=[train_loader_batch, val_loader])


if __name__ == '__main__':
    train()
