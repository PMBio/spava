##
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from tqdm.auto import tqdm
from typing import Union
from datasets.loaders.imc_data_loaders import get_cells_data_loader
from analyses.vae_expression.vae_expression_model import VAE
import anndata as ad
import scanpy as sc
from analyses.analisys_utils import compute_knn, louvain_plot
from datasets.imc_data_transform_utils import Prediction, Space

import os
from utils import reproducible_random_choice, get_execute_function, memory

e_ = get_execute_function()
# os.environ['SPATIALMUON_NOTEBOOK'] = 'analyses/vae_expression/vae_expression_analysis.py'

##
# re-train the best model but by perturbing the dataset
# if True:
if False:
    from analyses.vae_expression.vae_expression_model import objective, ppp
    from utils import file_path

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = "vae_expression"
    ppp.PERTURB = True
    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///" + file_path("optuna_vae_expression.sqlite"),
    )
    print('best trial:')
    print(study.best_trial)
    objective(study.best_trial)
    sys.exit(0)

##
if e_():
    SPLIT = "validation"
    loader_non_perturbed = get_cells_data_loader(split=SPLIT, batch_size=1024)
    loader_perturbed = get_cells_data_loader(split=SPLIT, batch_size=1024, perturb=True)

##
if e_():
    n = len(loader_non_perturbed.dataset)
    random_indices = reproducible_random_choice(n, 10000)

##
if e_():
    MODEL_CHECKPOINT = (
        "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/expression_vae/version_51"
        "/checkpoints/last.ckpt"
    )
#
def precompute(loader, expression_model_checkpoint, random_indices):
    expression_vae = VAE.load_from_checkpoint(expression_model_checkpoint)
    print("merging expressions and computing embeddings... ", end="")
    all_mu = []
    all_expression = []
    for data in tqdm(loader, desc="embedding expression"):
        raster, mask, expression, is_corrupted = data
        a, b, mu, std, z = expression_vae(expression)
        all_mu.append(mu)
        all_expression.append(expression)
    mus = torch.cat(all_mu, dim=0)
    expressions = torch.cat(all_expression, dim=0)
    print("done")

    a0 = ad.AnnData(mus.detach().numpy())
    sc.tl.pca(a0)
    sc.pl.pca(a0)
    a1 = ad.AnnData(expressions.numpy())

    b0 = a0[random_indices]
    b1 = a1[random_indices]
    # a, b = a0, b0
    # for a, b in tqdm(zip([a0, a1], [b0, b1]), desc="AnnData objects", total=2):
    for b in tqdm([b0, b1], desc="AnnData objects", total=2):
        print("computing umap... ", end="")
        sc.pp.neighbors(b)
        sc.tl.umap(b)
        sc.tl.louvain(b)
        print("done")

        print("computing nearest neighbors on subsetted data... ", end="")
        compute_knn(b)

        # print("computing nearest neighbors on the full data... ", end="")
        # nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(a.X)
        # distances, indices = nbrs.kneighbors(a.X)
        # a.obsm["nearest_neighbors"] = indices
        # print("done")
    # return a0, a1, b0, b1
    return b0, b1


if e_():
    b0, b1 = precompute(
        loader=loader_perturbed,
        expression_model_checkpoint=MODEL_CHECKPOINT,
        random_indices=random_indices,
    )
##
if e_():
    louvain_plot(b1, "expression (perturbed)")
    louvain_plot(b0, "latent (perturbed)")

##
if e_():
    data = loader_non_perturbed.__iter__().__next__()
    data_non_perturbed = loader_non_perturbed.__iter__().__next__()
    i0, i1 = torch.where(data[-1] == 1)
    # perturbed data are zero, non perturbed data are ok
    print(data[0][i0, i1])
    print(data_non_perturbed[0][i0, i1])
    # just a hash
    h = np.sum(np.concatenate(np.where(loader_perturbed.dataset.corrupted_entries == 1)))
    print(
        "corrupted entries hash:",
        h,
    )

##
if e_():
    expression_vae = VAE.load_from_checkpoint(MODEL_CHECKPOINT)

    debug_i = 0
    print("merging expressions and computing embeddings... ", end="")
    all_mu = []
    all_expression = []
    all_a = []
    all_b = []
    all_is_perturbed = []
    all_expression_non_perturbed = []
    for data, data_non_perturbed in tqdm(
        zip(loader_perturbed, loader_non_perturbed),
        desc="embedding expression",
        total=len(loader_perturbed),
    ):
        _, _, expression, is_perturbed = data
        _, _, expression_non_perturbed, _ = data_non_perturbed
        a, b, mu, std, z = expression_vae(expression)
        all_mu.append(mu)
        all_expression.append(expression)
        all_a.append(a)
        all_b.append(b)
        all_is_perturbed.append(is_perturbed)
        all_expression_non_perturbed.append(expression_non_perturbed)
        if debug_i < 5:
            # perturbed entries
            i0, i1 = torch.where(data[-1] == 1)
            # non perturbed entries
            j0, j1 = torch.where(data[-1] == 0)
            assert torch.isclose(torch.sum(expression[i0, i1]), torch.tensor([0.0]))
            assert torch.all(expression[j0, j1] == expression_non_perturbed[j0, j1])
        debug_i += 1

    mus = torch.cat(all_mu, dim=0)
    expressions = torch.cat(all_expression, dim=0)
    expressions_non_perturbed = torch.cat(all_expression_non_perturbed, dim=0)
    a_s = torch.cat(all_a, dim=0)
    b_s = torch.cat(all_b, dim=0)
    are_perturbed = torch.cat(all_is_perturbed, dim=0)
    reconstructed = expression_vae.expected_value(a_s, b_s)

##
# if m:
#     n_channels = expressions.shape[1]
#
#     original_non_perturbed = []
#     reconstructed_zero = []
#     for i in range(n_channels):
#         x = expressions_non_perturbed[:, i][are_perturbed[:, i]]
#         original_non_perturbed.append(x)
#         x = reconstructed[:, i][are_perturbed[:, i]]
#         reconstructed_zero.append(x)

##
# if e_():
if True:
    kwargs = dict(
        original=expressions_non_perturbed.cpu().numpy(),
        corrupted_entries=are_perturbed.cpu().numpy(),
        predictions_from_perturbed=reconstructed.detach().cpu().numpy(),
        space=Space.scaled_mean.value,
        name="expression vae",
        split="validation",
    )
    vae_predictions = Prediction(**kwargs)

    plt.style.use('default')
    vae_predictions.plot_reconstruction()
    # vae_predictions.plot_scores()
    vae_predictions.plot_summary()


##
if e_():
    # this seems to be broken
    p = vae_predictions.transform_to(Space.raw_sum)
    p.name = "expression vae raw"
    p.plot_reconstruction()
    # p.plot_scores()

##
# from old_code.data2 import file_path
# import pickle
#
# if m:
#     d = {"vanilla VAE": kwargs}
#     pickle.dump(d, open(file_path("ah_scores.pickle"), "wb"))
