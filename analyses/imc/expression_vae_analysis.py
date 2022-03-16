##
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from tqdm.auto import tqdm
from typing import Union
from datasets.loaders.imc_loaders import get_cells_data_loader
import anndata as ad
import scanpy as sc
from analyses.analisys_utils import compute_knn, louvain_plot
from datasets.imc_transform_utils import IMCPrediction, Space
import colorama

import os
from utils import reproducible_random_choice, get_execute_function

e_ = get_execute_function()
# os.environ["SPATIALMUON_TEST"] = "analyses/imc/expression_vae_analysis.py"
# os.environ["SPATIALMUON_NOTEBOOK"] = "analyses/imc/expression_vae_analysis.py"

if "SPATIALMUON_FLAGS" in os.environ:
    SPATIALMUON_FLAGS = os.environ["SPATIALMUON_FLAGS"]
else:
    # SPATIALMUON_FLAGS = "expression_vae"
    # SPATIALMUON_FLAGS = "image_expression_vae"
    SPATIALMUON_FLAGS = "image_expression_conv_vae"

print(
    f"{colorama.Fore.MAGENTA}SPATIALMUON_FLAGS = {SPATIALMUON_FLAGS}{colorama.Fore.RESET}"
)


def is_expression_vae():
    return SPATIALMUON_FLAGS == "expression_vae"


def is_image_expression_vae():
    return SPATIALMUON_FLAGS == "image_expression_vae"


def is_image_expression_conv_vae():
    return SPATIALMUON_FLAGS == "image_expression_conv_vae"


torch.multiprocessing.set_sharing_strategy("file_system")


assert (
    np.sum(
        [is_expression_vae(), is_image_expression_vae(), is_image_expression_conv_vae()]
    )
    == 1
)

MODEL_NAME = SPATIALMUON_FLAGS

##
if is_expression_vae():
    from models.expression_vae import VAE
elif is_image_expression_vae():
    from models.image_expression_vae import VAE
elif is_image_expression_conv_vae():
    from models.image_expression_conv_vae import VAE
else:
    assert False
##
if e_():
    from analyses.imc.expression_vae_runner import objective, ppp
    from utils import file_path

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = f"imc_{MODEL_NAME}"

    # ppp.PERTURB = True
    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///" + file_path(f"optuna_{study_name}.sqlite"),
    )
    print("best trial:")
    print(study.best_trial)
    print(study.best_trial.user_attrs["version"])

    # re-train the best model but by perturbing the dataset
    if False:
        if True:
            objective(study.best_trial)
            sys.exit(0)
        else:
            # manually update version from the just trained perturbed model
            if is_expression_vae():
                pass
            elif is_image_expression_vae():
                pass
            elif is_image_expression_conv_vae():
                pass
            else:
                assert False
            version = -1
    else:
        version = study.best_trial.user_attrs["version"]

##
if e_():
    SPLIT = "validation"
    if is_expression_vae():
        batch_size = 1024
        num_workers = 10
    elif is_image_expression_vae() or is_image_expression_conv_vae():
        batch_size = 128
        num_workers = 10
    else:
        assert False
    kwargs = {}
    if is_image_expression_conv_vae():
        kwargs = {'pca_tiles': True}
    loader_non_perturbed = get_cells_data_loader(
        split=SPLIT, batch_size=batch_size, num_workers=num_workers, **kwargs
    )
    loader_perturbed = get_cells_data_loader(
        split=SPLIT, batch_size=batch_size, perturb=True, num_workers=num_workers, **kwargs
    )

##
if e_():
    n = len(loader_non_perturbed.dataset)
    if "SPATIALMUON_TEST" in os.environ:
        random_indices = reproducible_random_choice(n, n - 1)
    else:
        random_indices = reproducible_random_choice(n, 10000)

##
if e_():
    MODEL_CHECKPOINT = file_path(
        f"checkpoints/{study_name}/version_{version}/checkpoints/last.ckpt"
    )
#
def precompute(loader, expression_model_checkpoint, random_indices):
    expression_vae = VAE.load_from_checkpoint(expression_model_checkpoint)
    print("merging expressions and computing embeddings... ", end="")
    all_mu = []
    all_expression = []
    for data in tqdm(loader, "embedding"):
        if is_expression_vae():
            raster, mask, expression, is_corrupted = data
            a, b, mu, std, z = expression_vae(expression)
        elif is_image_expression_vae() or is_image_expression_conv_vae():
            image_input, expression, is_corrupted = expression_vae.unfold_batch(data)
            a, b, mu, std, z = expression_vae(image_input)
        else:
            assert False
        all_mu.append(mu.detach())
        all_expression.append(expression.detach())
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
        if is_expression_vae():
            _, _, expression, is_perturbed = data
            _, _, expression_non_perturbed, _ = data_non_perturbed
            a, b, mu, std, z = expression_vae(expression)
        elif is_image_expression_vae() or is_image_expression_conv_vae():
            image_input, expression, is_perturbed = expression_vae.unfold_batch(data)
            _, expression_non_perturbed, _ = expression_vae.unfold_batch(
                data_non_perturbed
            )
            a, b, mu, std, z = expression_vae(image_input)
        else:
            assert False
        all_mu.append(mu.detach().numpy())
        all_expression.append(expression)
        all_a.append(a.detach().numpy())
        all_b.append(b.detach().numpy())
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

    mus = np.concatenate(all_mu, axis=0)
    expressions = np.concatenate(all_expression, axis=0)
    expressions_non_perturbed = np.concatenate(all_expression_non_perturbed, axis=0)
    a_s = np.concatenate(all_a, axis=0)
    b_s = np.concatenate(all_b, axis=0)
    are_perturbed = np.concatenate(all_is_perturbed, axis=0)
    reconstructed = expression_vae.expected_value(
        torch.tensor(a_s), torch.tensor(b_s)
    ).numpy()

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
if e_():
    # if True:
    kwargs = dict(
        original=expressions_non_perturbed,
        corrupted_entries=are_perturbed,
        predictions_from_perturbed=reconstructed,
        space=Space.scaled_mean.value,
        name=f"{MODEL_NAME} expression vae",
        split="validation",
    )
    vae_predictions = IMCPrediction(**kwargs)

    plt.style.use("default")
    vae_predictions.plot_reconstruction()
    # vae_predictions.plot_scores()
    vae_predictions.plot_summary()


##
if e_():
    # this seems to be broken
    p = vae_predictions.transform_to(Space.raw_sum)
    p.name = f"{MODEL_NAME} expression vae raw"
    p.plot_reconstruction()
    # p.plot_scores()

##
import pickle

if e_():
    f = file_path("imc/imputation_scores")
    os.makedirs(f, exist_ok=True)

    d = {"vanilla VAE": kwargs}
    pickle.dump(
        d,
        open(
            file_path(f"imc/imputation_scores/expression_vae_{MODEL_NAME}.pickle"), "wb"
        ),
    )
