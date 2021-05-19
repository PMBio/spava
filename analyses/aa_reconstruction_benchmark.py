#%%
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

#%%
from models.ah_expression_vaes_lightning import PerturbedCellDataset
plt.figure()
plt.scatter([0, 1], [0, 1])
plt.show()

#%%
from models.ah_expression_vaes_lightning import PerturbedCellDataset
plt.figure()
plt.scatter([0, 1], [0, 1])
plt.show()