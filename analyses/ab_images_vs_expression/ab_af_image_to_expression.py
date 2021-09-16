##
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import matplotlib.cm

from data2 import PerturbedRGBCells, PerturbedCellDataset, IndexInfo
import matplotlib.pyplot as plt
from models.ah_expression_vaes_lightning import VAE as ExpressionVAE, get_loaders
import scanpy as sc
import anndata as ad
import seaborn as sns
import pandas as pd
import optuna
from analyses.essentials import *
from utils import memory

m = __name__ == "__main__"

##
if m:
    SPLIT = "validation"
    ds_original = PerturbedRGBCells(split=SPLIT)
    ds_perturbed = PerturbedRGBCells(split=SPLIT)
    ds_perturbed.perturb()

##
if m:
    from models.aj_image_expression import VAE as ImageToExpressionVAE

    # decent models:
    # 49 non perturbed, 63 perturbed
    # 94 non perturbed, <does not train> perturbed
    # 75 non perturbed, <does not train> perturbed
    # 87 non perturbed, 99 perturbed: bad embedding
    # 78 non perturbed
    f_original = (
        "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/image_to_expression/version_78"
        "/checkpoints/last.ckpt"
    )
    model_original = ImageToExpressionVAE.load_from_checkpoint(f_original)
    model_original.cuda()
    loader_original = DataLoader(
        ds_original, batch_size=1024, num_workers=8, pin_memory=True
    )

    f_perturbed = (
        "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/image_to_expression/version_99"
        "/checkpoints/last.ckpt"
    )
    model_perturbed = ImageToExpressionVAE.load_from_checkpoint(f_perturbed)
    model_perturbed.cuda()
    loader_perturbed = DataLoader(
        ds_perturbed, batch_size=1024, num_workers=8, pin_memory=True
    )
##
if m:
    list_of_z = []
    with torch.no_grad():
        for data in tqdm(loader_original, desc="forwarding"):
            data = [d.to(model_original.device) for d in data]
            _, x, mask, _ = data
            z = [zz.cpu() for zz in model_original(x, mask)]
            list_of_z.append(z)
    torch.cuda.empty_cache()

##
if m:
    l = []
    for zz in list_of_z:
        a, mu, std, z = zz
        l.append(mu)
        # reconstructed = model_original.get_dist(a).mean
    mus = torch.cat(l, dim=0).numpy()

##
from analyses.essentials import scanpy_compute
import scanpy as sc
import anndata as ad

a = ad.AnnData(mus)
sc.tl.pca(a)
sc.pl.pca(a)
##
from utils import reproducible_random_choice
random_indices = reproducible_random_choice(len(a), 10000)

##
b = a[random_indices]

##
scanpy_compute(b)

##
sc.pl.umap(b, color='louvain')

##
if m:
    list_of_z_perturbed = []
    with torch.no_grad():
        for data in tqdm(loader_perturbed, desc="forwarding"):
            data = [d.to(model_perturbed.device) for d in data]
            _, x, mask, _ = data
            z = [zz.cpu() for zz in model_perturbed(x, mask)]
            list_of_z_perturbed.append(z)
    torch.cuda.empty_cache()

##

if m:
    l = []
    for zz in list_of_z_perturbed:
        a, mu, std, z = zz
        xx = model_original.get_dist(a).mean
        l.append(xx)
    reconstructed = torch.cat(l, dim=0).numpy()

##
if m:
    l = []
    for data in tqdm(loader_original, desc='merging expression'):
        expression, _, _, _ = data
        l.append(expression)
    expressions = torch.cat(l, dim=0).numpy()

    l = []
    for data in tqdm(loader_perturbed, desc='merging perturbed entries'):
        _, _, _, is_perturbed = data
        l.append(is_perturbed)
    are_perturbed = torch.cat(l, dim=0).numpy()

##
h = np.sum(np.concatenate(np.where(are_perturbed == 1)))
print('corrupted entries hash:', h)


##
if m:
    p = Prediction(original=expressions,
               corrupted_entries=are_perturbed,
               predictions_from_perturbed=reconstructed,
               space=Space.scaled_mean,
               name='image to expression',
               split='validation')
    p.plot_reconstruction()
    p.plot_scores()
