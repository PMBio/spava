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


##
v = x[a].detach().numpy()
w = x_zero_pred_mean[a].detach().numpy()
axes = plt.subplots(1, 2, figsize=(10, 5))[1]
axes[0].hist(v, bins=30)
axes[0].set(xlim=(0, 5))
axes[1].hist(w, bins=5)
axes[1].set(xlim=(0, 5))
plt.show()
##
def plot_imputation(imputed, original, xtext): #, zeros, i, j, ix, xtext):
    # all_index = i[ix], j[ix]
    # x, y = imputed[all_index], original[all_index]
    #
    # x = x[zeros[all_index] == 0]
    # y = y[zeros[all_index] == 0]
    #
    ymax = 4
    # mask = x < ymax
    # x = x[mask]
    # y = y[mask]
    #
    # mask = y < ymax
    # x = x[mask]
    # y = y[mask]
    #
    # l = np.minimum(x.shape[0], y.shape[0])
    #
    # x = x[:l]
    # y = y[:l]

    # data = np.vstack([x, y])
    data = np.vstack([imputed, original])

    plt.figure(figsize=(5, 5))

    axes = plt.gca()
    axes.set_xlim([0, ymax])
    axes.set_ylim([0, ymax])

    nbins = 50

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data)
    xi, yi = np.mgrid[0:ymax:nbins * 1j, 0:ymax:nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.title(xtext, fontsize=12)
    plt.ylabel("Imputed counts")
    plt.xlabel('Original counts')

    plt.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap="Reds")

    a, _, _, _ = np.linalg.lstsq(original[:, np.newaxis], imputed)
    l = np.linspace(0, ymax)
    plt.plot(l, a * l, color='black')

    plt.plot(l, l, color='black', linestyle=":")
    plt.show()

plot_imputation(w, v, 'gaussian noise model')