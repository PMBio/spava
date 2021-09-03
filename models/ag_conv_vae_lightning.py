# aaa = True
import numpy as np


class Ppp:
    pass


ppp = Ppp()
if 'aaa' in locals():
    ppp.DEBUG_TORCH = 'yessss'
ppp.MAX_EPOCHS = 20
# ppp.COOL_CHANNELS = np.array([38, 38, 38])
ppp.COOL_CHANNELS = np.arange(39)
ppp.BATCH_SIZE = 1024
ppp.LEARNING_RATE = 0.8e-3
ppp.VAE_BETA = 100
# ppp.DEBUG = True
ppp.DEBUG = False
if ppp.DEBUG and not 'DEBUG_TORCH' in ppp.__dict__:
    ppp.NUM_WORKERS = 0
    ppp.DETECT_ANOMALY = True
else:
    if 'DEBUG_TORCH' in ppp.__dict__:
        ppp.NUM_WORKERS = 0
    else:
        ppp.NUM_WORKERS = 16
    ppp.DETECT_ANOMALY = False


# ppp.NOISE_MODEL = 'gaussian'
# ppp.NOISE_MODEL = 'nb'

def set_ppp_from_loaded_model(pl_module):
    global ppp
    ppp = pl_module.hparams


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(1234)

from torch import nn

from models.ag_resnet_vae import resnet_encoder, resnet_decoder
# from pl_bolts.models.autoencoders.components import (
# resnet18_decoder,
# resnet18_encoder,
# )
from argparse import ArgumentParser

# matplotlib.use('Agg')
from torchvision.utils import make_grid
import PIL
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from torch import autograd
import contextlib
import torch
import pyro

# from pl_bolts.utils import _TORCHVISION_AVAILABLE
# from pl_bolts.utils.warnings import warn_missing_pkg

import torchvision.transforms
from data2 import PerturbedRGBCells

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
                                        2.0277, 3.3281, 3.9273])[ppp.COOL_CHANNELS]


# print('fino a qui tutto bene')

def get_image(loader, model, return_cells=False):
    all_originals = []
    all_reconstructed = []
    all_masks = []
    mask_color = torch.tensor([x / 255 for x in [254, 112, 31]]).float()
    red_color = torch.tensor([x / 255 for x in [255, 0, 0]]).float()
    new_size = (128, 128)
    upscale = torchvision.transforms.Resize(new_size, interpolation=PIL.Image.NEAREST)
    n = 15
    with torch.no_grad():
        batch = loader.__iter__().__next__()
        omes = batch[0]
        masks = batch[1]
        if len(batch) == 3:
            perturbed_entries = batch[2]
        else:
            perturbed_entries = None
        assert len(omes.shape) == 4
        assert len(omes) >= n
        data = omes[:n].to(model.device)
        masks_data = masks[:n].to(model.device)
        alpha_pred, beta_pred, mu, std, z = model.forward(data, masks_data)
    n_channels = data.shape[1]
    # I'm lazy
    full_mask = torch.tensor([[mask_color.tolist() for _ in range(n_channels)] for _ in range(n_channels)])
    full_mask = upscale(full_mask.permute(2, 0, 1))
    all_original_c = {c: [] for c in range(n_channels)}
    all_original_masked_c = {c: [] for c in range(n_channels)}
    all_reconstructed_c = {c: [] for c in range(n_channels)}
    all_reconstructed_masked_c = {c: [] for c in range(n_channels)}

    for i in range(n):
        original = data[i].cpu().permute(1, 2, 0) * quantiles_for_normalization
        alpha = alpha_pred[i].cpu().permute(1, 2, 0)
        beta = beta_pred[i].cpu().permute(1, 2, 0)
        dist = model.get_dist(alpha, beta)
        mean = dist.mean
        # r = alpha
        # p = 1 / (1 + beta)
        # # r_hat = pred[i].cpu().permute(1, 2, 0)
        # # p = torch.sigmoid(model.negative_binomial_p_logit).cpu().detach()
        # # mean = model.negative_binomial_mean(r=r_hat, p=p)
        # mean = p * r / (1 - p)
        reconstructed = mean * quantiles_for_normalization

        a_original = original.amin(dim=(0, 1))
        b_original = original.amax(dim=(0, 1))
        m = masks_data[i].cpu().bool()
        mm = torch.squeeze(m, 0)
        reconstructed_flattened = torch.reshape(reconstructed, (-1, reconstructed.shape[-1]))
        mask_flattened = mm.flatten()
        if mask_flattened.sum() > 0:
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

            all_masks.append(mm)
            #### original_masked = original.clone()
            #### original_masked[mm_not, :] = mask_color
            #### reconstructed_masked = reconstructed.clone()
            #### reconstructed_masked[mm_not, :] = mask_color

            for c in range(n_channels):
                is_perturbed_entry = perturbed_entries is not None and perturbed_entries[i, c]
                original_c = original[:, :, c]
                original_c = torch.stack([original_c] * 3, dim=2)

                reconstructed_c = reconstructed[:, :, c]
                reconstructed_c = torch.stack([reconstructed_c] * 3, dim=2)

                def f(t):
                    t = t.permute(2, 0, 1)
                    t = upscale(t)
                    return t

                def overlay_mask(t, is_perturbed_entry=False):
                    t = t.clone()
                    if not is_perturbed_entry:
                        color = mask_color
                    else:
                        color = red_color
                    t[mm_not, :] = color
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
                all_original_masked_c[c].append(f(overlay_mask(t, is_perturbed_entry)))
                t = (reconstructed_c - a_c) / (b_c - a_c)
                all_reconstructed_c[c].append(f(t))
                all_reconstructed_masked_c[c].append(f(overlay_mask(t, is_perturbed_entry)))

            original = upscale(original.permute(2, 0, 1))
            reconstructed = upscale(reconstructed.permute(2, 0, 1))
            #### original_masked = upscale(original_masked.permute(2, 0, 1))
            #### reconstructed_masked = upscale(reconstructed_masked.permute(2, 0, 1))

            all_originals.append(original)
            all_reconstructed.append(reconstructed)
            #### all_originals_masked.append(original_masked)
            #### all_reconstructed_masked.append(reconstructed_masked)
        else:
            all_originals.append(upscale(original.permute(2, 0, 1)))
            all_reconstructed.append(upscale(reconstructed.permute(2, 0, 1)))
            all_masks.append(torch.tensor(np.zeros(new_size, dtype=bool)))
            for c in range(n_channels):
                all_original_c[c].append(full_mask)
                all_reconstructed_c[c].append(full_mask)
                all_original_masked_c[c].append(full_mask)
                all_reconstructed_masked_c[c].append(full_mask)

    l = []  ####
    pixels = []
    for original, reconstructed, mask in zip(all_originals, all_reconstructed, all_masks):
        pixels.extend(original.permute(1, 2, 0).reshape((-1, n_channels)))
        upscaled_mask = upscale(torch.unsqueeze(mask, 0))
        upscaled_mask = torch.squeeze(upscaled_mask, 0).bool()
        masked_reconstructed = reconstructed[:, upscaled_mask]
        pixels.extend(masked_reconstructed.permute(1, 0))
    from sklearn.decomposition import PCA
    all_pixels = torch.stack(pixels).numpy()
    reducer = PCA(3)
    reducer.fit(all_pixels)
    transformed = reducer.transform(all_pixels)
    a = np.min(transformed, axis=0)
    b = np.max(transformed, axis=0)
    all_originals_pca = []
    all_reconstructed_pca = []
    all_originals_pca_masked = []
    all_reconstructed_pca_masked = []

    def scale(x, a, b):
        x = np.transpose(x, (1, 2, 0))
        x = (x - a) / (b - a)
        x = np.transpose(x, (2, 0, 1))
        return x

    for original, reconstructed, mask in zip(all_originals, all_reconstructed, all_masks):
        to_transform = original.permute(1, 2, 0).reshape((-1, n_channels)).numpy()
        pca = reducer.transform(to_transform)
        pca.shape = (original.shape[1], original.shape[2], 3)
        original_pca = np.transpose(pca, (2, 0, 1))
        original_pca = scale(original_pca, a, b)
        all_originals_pca.append(torch.tensor(original_pca))

        to_transform = reconstructed.permute(1, 2, 0).reshape((-1, n_channels)).numpy()
        pca = reducer.transform(to_transform)
        pca.shape = (reconstructed.shape[1], reconstructed.shape[2], 3)
        reconstructed_pca = np.transpose(pca, (2, 0, 1))
        reconstructed_pca = scale(reconstructed_pca, a, b)
        all_reconstructed_pca.append(torch.tensor(reconstructed_pca))

        mask = torch.logical_not(mask)
        upscaled_mask = upscale(torch.unsqueeze(mask, 0))
        upscaled_mask = torch.squeeze(upscaled_mask, 0).bool()

        original_pca_masked = original_pca.copy()
        original_pca_masked = original_pca_masked.transpose((1, 2, 0))
        original_pca_masked[upscaled_mask, :] = mask_color
        original_pca_masked = original_pca_masked.transpose((2, 0, 1))
        all_originals_pca_masked.append(torch.tensor(original_pca_masked))

        reconstructed_pca_masked = reconstructed_pca.copy()
        reconstructed_pca_masked = reconstructed_pca_masked.transpose((1, 2, 0))
        reconstructed_pca_masked[upscaled_mask, :] = mask_color
        reconstructed_pca_masked = reconstructed_pca_masked.transpose((2, 0, 1))
        all_reconstructed_pca_masked.append(torch.tensor(reconstructed_pca_masked))

    l.extend(all_originals_pca + all_reconstructed_pca + all_originals_pca_masked + all_reconstructed_pca_masked)

    for c in range(n_channels):
        l += (all_original_c[c] + all_reconstructed_c[c] + all_original_masked_c[c] + all_reconstructed_masked_c[c])

    #### from sklearn.decomposition import PCA
    #### reducer = PCA(3)

    img = make_grid(l, nrow=n)
    if not return_cells:
        return img
    else:
        return img, all_original_c, all_reconstructed_c, all_masks


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
        # a = pl_module.negative_binomial_p_logit
        # b = pl_module.boosted_sigmoid(a)
        # trainer.logger.experiment.add_scalars(f'negative_binomial_p_logit', {f'channel{i}': a[i] for i in range(len(
        #     a))}, trainer.global_step)
        # trainer.logger.experiment.add_scalars(f'negative_binomial_p', {f'channel{i}': b[i] for i in range(len(b))},
        #                                       trainer.global_step)

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
    if ppp.DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


class VAE(pl.LightningModule):
    def __init__(self, n_channels, enc_out_dim=256, latent_dim=64, input_height=32, **kwargs):
        super().__init__()

        # self.save_hyperparameters(kwargs)
        self.save_hyperparameters()

        # encoder, decoder
        self.n_channels = n_channels
        self.encoder = resnet_encoder(first_conv=False, maxpool1=False, n_channels=self.n_channels)
        self.decoder = resnet_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False,
            n_channels=self.n_channels
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)
        self.softplus = nn.Softplus()

        # for the gaussian likelihood
        # self.log_scale = nn.Parameter(torch.Tensor([0.4]))

        # value such that if we apply the sigmoid function we get the p parameter of a negative binomial,
        # one per channel

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=ppp.LEARNING_RATE)

    def get_dist(self, alpha, beta):
        return pyro.distributions.GammaPoisson(alpha, beta)

    def reconstruction_likelihood(self, alpha, beta, x, mask, corrupted_entries):
        dist = self.get_dist(alpha, beta)
        log_pxz = dist.log_prob(x)

        # measure prob of seeing image under p(x|z)
        # log_pxz = dist.log_prob(x)
        if mask is None:
            mask = torch.ones_like(log_pxz)
        log_pxz[corrupted_entries, :, :] = 0.
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
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def loss_function(self, x, alpha, beta, mu, std, z, mask, corrupted_entries):
        # reconstruction loss
        # print(x_hat.shape)
        cm = get_detect_anomaly_cm()
        with cm:
            recon_loss = self.reconstruction_likelihood(alpha, beta, x, mask, corrupted_entries)
            # kl
            kl = self.kl_divergence(z, mu, std)
            # elbo
            elbo = (ppp.VAE_BETA * kl - recon_loss)
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
            log_alpha, log_beta = self.decoder(z, mask)
            alpha = torch.exp(log_alpha)
            beta = torch.exp(log_beta)
            return alpha, beta, mu, std, z

    def training_step(self, batch, batch_idx):
        # print('min, max:', batch.min().cpu().detach(), batch.max().cpu().detach())
        x = batch[0]
        mask = batch[1]
        corrupted_entries = batch[2]
        # encode x to get the mu and variance parameters
        alpha, beta, mu, std, z = self.forward(x, mask)
        elbo, kl, recon_loss = self.loss_function(x, alpha, beta, mu, std, z, mask, corrupted_entries)

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'reconstruction': recon_loss.mean(),
        })

        return elbo

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x = batch[0]
        mask = batch[1]
        corrupted_entries = batch[2]
        alpha, beta, mu, std, z = self.forward(x, mask)
        elbo, kl, recon_loss = self.loss_function(x, alpha, beta, mu, std, z, mask, corrupted_entries)

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

    def on_post_move_to_device(self):
        self.decoder.mask_conv1.weight = self.encoder.mask_conv1.weight
        self.decoder.mask_conv2.weight = self.encoder.mask_conv2.weight
        self.decoder.mask_conv1x1.weight = self.encoder.mask_conv1x1.weight
        print('sharing the weights between encoder and decoder for the convnet operating on the mask')


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


# maybe use inheritance


def train(perturb=False):
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    train_ds = PerturbedRGBCells('train', augment=True, aggressive_rotation=True)
    train_ds_validation = PerturbedRGBCells('train')
    val_ds = PerturbedRGBCells('validation')

    if perturb:
        train_ds.perturb()
        train_ds_validation.perturb()
        val_ds.perturb()
    from data2 import file_path
    logger = TensorBoardLogger(save_dir=file_path('checkpoints'), name='resnet_vae')
    print(f'logging in {logger.experiment.log_dir}')
    checkpoint_callback = ModelCheckpoint(dirpath=file_path(f'{logger.experiment.log_dir}/checkpoints'),
                                          monitor='elbo',
                                          # every_n_train_steps=2,
                                          save_last=True,
                                          save_top_k=1)
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=ppp.MAX_EPOCHS, callbacks=[
        ImageSampler(), LogComputationalGraph(), checkpoint_callback
    ],
                         logger=logger,
                         log_every_n_steps=15, val_check_interval=2 if ppp.DEBUG else 50)
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
    vae = VAE(n_channels=len(ppp.COOL_CHANNELS), **ppp.__dict__)
    trainer.fit(vae, train_dataloader=train_loader, val_dataloaders=[train_loader_batch, val_loader])
    print(f'finished logging in {logger.experiment.log_dir}')


if __name__ == '__main__':
    train(perturb=True)
