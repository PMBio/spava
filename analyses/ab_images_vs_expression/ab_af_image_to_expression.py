##
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import matplotlib.cm

from data2 import PerturbedRGBCells, PerturbedCellDataset, IndexInfo
from graphs import CellExpressionGraphOptimized
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

#
if m:
    SPLIT = "validation"
    # MODEL = 'conv_encoder'
    # MODEL = "resnet_encoder"
    MODEL = "ai_gnn_vae"
    assert MODEL in ["conv_encoder", "resnet_encoder", "ai_gnn_vae"]
    from models.aj_image_expression import VAE as ImageToExpressionVAE
    from models.ak_resnet_expression import ResNetToExpression
    from models.ai_gnn_vae import GnnVae

    # best models
    # conv vae:
    #     49 non perturbed, 63 perturbed
    #     116 non perturbed, 151 perturbed
    # resnet vae:
    #     154 non perturbed, 162 perturbed
    # gnn vae:
    #     94 non perturbed,

    if MODEL == "conv_encoder":
        model_index_non_perturbed = 116
        model_index_perturbed = 151
        tensorboard_label = "image_to_expression"
        model = ImageToExpressionVAE
        ds_original = PerturbedRGBCells(split=SPLIT)
        ds_perturbed = PerturbedRGBCells(split=SPLIT)
        ds_perturbed.perturb()
        data_loader_class = DataLoader
    elif MODEL == "resnet_encoder":
        model_index_non_perturbed = 154
        model_index_perturbed = 162
        model = ResNetToExpression
        tensorboard_label = "image_to_expression"
        ds_original = PerturbedRGBCells(split=SPLIT)
        ds_perturbed = PerturbedRGBCells(split=SPLIT)
        ds_perturbed.perturb()
        data_loader_class = DataLoader
    elif MODEL == "ai_gnn_vae":
        model_index_non_perturbed = 118
        model_index_perturbed = 118
        model = GnnVae
        tensorboard_label = "gnn_vae"
        ds_original = CellExpressionGraphOptimized(split=SPLIT, graph_method="gaussian")
        ds_perturbed = CellExpressionGraphOptimized(split=SPLIT, graph_method="gaussian", perturb=True)
        data_loader_class = GeometricDataLoader
    else:
        raise RuntimeError()

    f_original = (
        f"/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/{tensorboard_label}/version_"
        f"{model_index_non_perturbed}"
        "/checkpoints/last.ckpt"
    )
    model_original = model.load_from_checkpoint(f_original)
    model_original.cuda()
    loader_original = data_loader_class(
        ds_original, batch_size=1024, num_workers=8, pin_memory=True
    )

    f_perturbed = (
        f"/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/{tensorboard_label}/version_"
        f"{model_index_perturbed}"
        "/checkpoints/last.ckpt"
    )
    model_perturbed = model.load_from_checkpoint(f_perturbed)
    model_perturbed.cuda()
    loader_perturbed = data_loader_class(
        ds_perturbed, batch_size=1024, num_workers=8, pin_memory=True
    )
#

def get_list_of_z(loader, model):
    list_of_z = []
    with torch.no_grad():
        for data in tqdm(loader_original, desc="forwarding"):
            if MODEL == 'ai_gnn_vae':
                data.to(model_original.device)
                output = model_original(data.x, data.edge_index, data.edge_attr, data.is_center)
                z = [zz.cpu() for zz in output]
            else:
                data = [d.to(model_original.device) for d in data]
                _, x, mask, _ = data
                z = [zz.cpu() for zz in model_original(x, mask)]
            list_of_z.append(z)
    torch.cuda.empty_cache()
    return list_of_z


if m and False:
    list_of_z = get_list_of_z(loader=loader_original, model=model_original)

#
    l = []
    for zz in list_of_z:
        a, mu, std, z = zz
        l.append(mu)
        # reconstructed = model_original.get_dist(a).mean
    mus = torch.cat(l, dim=0).numpy()
#
    from analyses.essentials import scanpy_compute
    import scanpy as sc
    import anndata as ad

    a = ad.AnnData(mus)
    sc.tl.pca(a)
    sc.pl.pca(a)
    #
    from utils import reproducible_random_choice

    random_indices = reproducible_random_choice(len(a), 10000)

    #
    b = a[random_indices]

    #
    scanpy_compute(b)

    #
    sc.pl.umap(b, color="louvain")
##
# until_here_because_the_perturbed_is_not_trained
##
if m:
    list_of_z_perturbed = get_list_of_z(loader=loader_perturbed, model=model_perturbed)

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
    for data in tqdm(loader_original, desc="merging expression"):
        if MODEL == 'ai_gnn_vae':
            expression = data.x[torch.where(data.is_center == 1.)[0], :]
        else:
            expression, _, _, _ = data
        l.append(expression)
    expressions = torch.cat(l, dim=0).numpy()

    l = []
    for data in tqdm(loader_perturbed, desc="merging perturbed entries"):
        if MODEL == 'ai_gnn_vae':
            is_perturbed = data.is_perturbed[torch.where(data.is_center == 1.)[0], :]
        else:
            _, _, _, is_perturbed = data
        l.append(is_perturbed)
    are_perturbed = torch.cat(l, dim=0).numpy()

##
h = np.sum(np.concatenate(np.where(are_perturbed == 1)))
print("corrupted entries hash:", h)


##
if m:
    p = Prediction(
        original=expressions,
        corrupted_entries=are_perturbed,
        predictions_from_perturbed=reconstructed,
        space=Space.scaled_mean,
        name=f"{MODEL} to expression",
        split="validation",
    )
    p.plot_reconstruction()
    p.plot_scores()

    p_raw = p.transform_to(Space.raw_sum)
    p_raw.plot_reconstruction()
    p_raw.plot_scores()

##
if m:
    ce = ds_perturbed.cell_expression_graph.cell_ds.corrupted_entries
    h = torch.sum(torch.cat(torch.where(ce == 1)))
    print(
        "corrupted entries hash:",
        h,
    )