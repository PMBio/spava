##
import os
import os.path

import pytorch_lightning as pl
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

##
# import ipdb
# from models.ah_expression_vaes_lightning import train
#
# train(perturb=True)

##
from models.ah_expression_vaes_lightning import PerturbedCellDataset

ds0 = PerturbedCellDataset('train')
ds0.perturb()
ds1 = PerturbedCellDataset('train')
ds1.perturb()
assert torch.all(ds0.corrupted_entries == ds1.corrupted_entries)

##
checkpoint = '/data/l989o/data/basel_zurich/spatial_uzh_processed/a/checkpoints/expression_vae/version_13/checkpoints' \
             '/epoch=359-step=719.ckpt'

from models.ah_expression_vaes_lightning import VAE

model = VAE.load_from_checkpoint(checkpoint)
##
plt.figure()
plt.hist(torch.exp(model.log_c).detach().numpy())
plt.title('distribution of log(c) values')
plt.show()

##
x = ds0.original_merged
x_zero = ds0.merged
x_pred = model.forward(x)[0]
x_zero_pred = model.forward(x_zero)[0]

x_pred_mean = model.expected_value(x_pred)
x_zero_pred_mean = model.expected_value(x_zero_pred)

##
a = ds0.corrupted_entries
score = torch.median(torch.abs(x[a] - x_zero_pred_mean[a]))
print(score)
# x[ds0.corrupted_entries], x_zero
##
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import time

##
v = x[a].detach().numpy()
w = x_zero_pred_mean[a].detach().numpy()
axes = plt.subplots(1, 2, figsize=(10, 5))[1]
axes[0].hist(v, bins=30)
axes[0].set(title='histogram of original expression values')
axes[0].set(xlim=(0, 5))
axes[1].hist(w, bins=5)
axes[1].set(title='histogram of predicted expression values')
axes[1].set(xlim=(0, 5))
plt.suptitle('only indices for corrupted entries')
plt.show()


##
def plot_imputation(imputed, original, xtext):  # , zeros, i, j, ix, xtext):
    # all_index = i[ix], j[ix]
    # x, y = imputed[all_index], original[all_index]
    #
    # x = x[zeros[all_index] == 0]
    # y = y[zeros[all_index] == 0]
    #
    cutoff = 2
    mask = imputed < cutoff
    imputed = imputed[mask]
    original = original[mask]

    mask = original < cutoff
    imputed = imputed[mask]
    original = original[mask]

    l = np.minimum(imputed.shape[0], original.shape[0])

    assert len(imputed) == len(original)
    imputed = imputed[:l]
    original = original[:l]

    # data = np.vstack([x, y])
    data = np.vstack([imputed, original])

    plt.figure(figsize=(5, 5))

    axes = plt.gca()
    axes.set_xlim([0, cutoff])
    axes.set_ylim([0, cutoff])

    nbins = 50

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    xi, yi = np.mgrid[0:cutoff:nbins * 1j, 0:cutoff:nbins * 1j]

    start = time.time()
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    print(f'evaluating the kernel on the mesh: {time.time() - start}')

    plt.title(xtext, fontsize=12)
    plt.ylabel("Imputed counts")
    plt.xlabel('Original counts')

    plt.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap="Reds")

    a, _, _, _ = np.linalg.lstsq(original[:, np.newaxis], imputed)
    l = np.linspace(0, cutoff)
    plt.plot(l, a * l, color='black')

    A = np.vstack([original, np.ones(len(original))]).T
    aa, _, _, _ = np.linalg.lstsq(A, imputed)
    plt.plot(l, aa[0] * l + aa[1], color='red')

    plt.plot(l, l, color='black', linestyle=":")
    plt.show()


plot_imputation(w, v, 'gaussian noise model')

##
plt.figure()
plt.scatter(v, w, s=1, alpha=0.05)
plt.xlabel('original')
plt.ylabel('imputed')
ax = plt.gca()
# ax.set_aspect('equal')
ax.set(xlim=(0, 2), ylim=(0, 2))
plt.show()


##
plt.figure()
plt.hist(w, bins=1000)
plt.title('historam of imputed')
plt.show()


##
plt.figure()
plt.hist(v, bins=10000)
plt.xscale('log')
plt.title('historam of originals')
plt.show()

##

