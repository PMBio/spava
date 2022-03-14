##
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from tqdm.auto import tqdm
from typing import Union
from datasets.loaders.visium_mousebrain_loaders import get_cells_data_loader
from models.expression_vae import VAE
import anndata as ad
import scanpy as sc
from analyses.analisys_utils import compute_knn, louvain_plot, scanpy_compute
from analyses.imputation_score import Prediction

import os
from utils import reproducible_random_choice, get_execute_function, memory

e_ = get_execute_function()
# os.environ["SPATIALMUON_TEST"] = "analyses/vae_expression/vae_expression_analysis.py"
# os.environ[
#     "SPATIALMUON_NOTEBOOK"
# ] = "analyses/visium_mousebrain/visium_mousebrain_analysis.py"

##
if e_():
    from utils import file_path

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = "visium_mousebrain_expression"
    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///" + file_path("optuna_visium_mousebrain_expression.sqlite"),
    )
    print("best trial:")
    print(study.best_trial)
    print(study.best_trial.user_attrs["version"])

    # re-train the best model but by perturbing the dataset
    if False:
        # if True:
        if False:
            # if True:
            from analyses.visium_mousebrain.expression_vae_runner import objective

            trial = study.best_trial
            trial.set_user_attr("MAX_EPOCHS", 50)
            objective(trial)
            sys.exit(0)
        else:
            # manually update version from the just trained perturbed model
            version = 15
    else:
        version = study.best_trial.user_attrs["version"]

##
if e_():
    MODEL_CHECKPOINT = file_path(
        f"checkpoints/visium_mousebrain_expression_vae/version_{version}/checkpoints/last.ckpt"
    )
    expression_vae = VAE.load_from_checkpoint(MODEL_CHECKPOINT)


def get_latent_representation(loader, model):
    print("merging expressions and computing embeddings... ", end="")
    all_mu = []
    all_expression = []
    all_a = []
    all_b = []
    all_is_corrupted = []
    for data in tqdm(loader, desc="embedding expression"):
        expression, is_corrupted = data
        a, b, mu, std, z = model(expression)
        all_mu.append(mu)
        all_expression.append(expression)
        all_a.append(a)
        all_b.append(b)
        all_is_corrupted.append(is_corrupted)
    mus = torch.cat(all_mu, dim=0)
    expressions = torch.cat(all_expression, dim=0)
    aas = torch.cat(all_a, dim=0)
    bbs = torch.cat(all_b, dim=0)
    are_perturbed = torch.cat(all_is_corrupted, dim=0)

    mus = ad.AnnData(mus.detach().numpy())
    expressions = ad.AnnData(expressions.detach().numpy())

    reconstructed = model.expected_value(aas, bbs)
    return mus, expressions, reconstructed, are_perturbed


if e_():
    train_loader_non_perturbed = get_cells_data_loader(
        split="train", batch_size=1024, only_expression=True
    )

    val_loader_non_perturbed = get_cells_data_loader(
        split="validation", batch_size=1024, only_expression=True
    )
    val_loader_perturbed = get_cells_data_loader(
        split="validation", batch_size=1024, perturb=True, only_expression=True
    )

    test_loader_non_perturbed = get_cells_data_loader(
        split="test", batch_size=1024, only_expression=True
    )

    (
        mus_train_non_perturbed,
        expression_train_non_perturbed,
        _,
        _,
    ) = get_latent_representation(
        loader=train_loader_non_perturbed,
        model=expression_vae,
    )

    (
        mus_val_non_perturbed,
        expression_val_non_perturbed,
        _,
        _,
    ) = get_latent_representation(
        loader=val_loader_non_perturbed,
        model=expression_vae,
    )
    (
        mus_val_perturbed,
        expression_val_perturbed,
        val_perturbed_reconstructed,
        val_are_perturbed,
    ) = get_latent_representation(
        loader=val_loader_perturbed,
        model=expression_vae,
    )

    (
        mus_test_non_perturbed,
        expression_test_non_perturbed,
        _,
        _,
    ) = get_latent_representation(
        loader=test_loader_non_perturbed,
        model=expression_vae,
    )

##
if e_():
    scanpy_compute(expression_val_non_perturbed)
    scanpy_compute(mus_val_non_perturbed)

    louvain_plot(expression_val_non_perturbed, "expression val (non perturbed)")
    louvain_plot(mus_val_non_perturbed, "latent val (non perturbed)")

##
if e_():
    merged = ad.AnnData.concatenate(
        mus_train_non_perturbed,
        mus_val_non_perturbed,
        mus_test_non_perturbed,
        batch_categories=["train", "validation", "test"],
    )
    scanpy_compute(merged)
    lou = merged.obs["louvain"]

    from datasets.visium_mousebrain import get_split_indices, get_smu_file

    train_indices = get_split_indices("train")
    val_indices = get_split_indices("validation")
    test_indices = get_split_indices("test")

    louvain_train = lou.iloc[: len(train_indices)]
    louvain_val = lou.iloc[len(train_indices) : len(lou) - len(test_indices)]
    louvain_test = lou.iloc[len(lou) - len(test_indices) :]

    categories = lou.cat.categories.tolist() + ["Nein"]

    assert len(louvain_train) + len(louvain_val) + len(louvain_test) == len(lou)
    assert all([s.endswith("-train") for s in louvain_train.index.tolist()])
    assert all([s.endswith("-validation") for s in louvain_val.index.tolist()])
    assert all([s.endswith("-test") for s in louvain_test.index.tolist()])

    import pandas as pd

    ordered_lou = pd.Categorical(["Nein"] * len(lou), categories=categories)

    ordered_lou[train_indices] = louvain_train.to_numpy()
    ordered_lou[val_indices] = louvain_val.to_numpy()
    ordered_lou[test_indices] = louvain_test.to_numpy()

    assert ordered_lou.value_counts()["Nein"] == 0
    ordered_lou = ordered_lou.remove_categories("Nein")
    lou_for_smu = ordered_lou.astype("category")

    s = get_smu_file(read_only=False)
    s["visium"]["processed"].obs["vae"] = lou_for_smu
    s["visium"]["processed"].masks.obj_has_changed("obs")
    s.commit_changes_on_disk()
    s.backing.close()

##
if e_():
    s = get_smu_file(read_only=False)
    _, ax = plt.subplots(1)
    s["visium"]["image"].plot(ax=ax)
    s["visium"]["processed"].masks.plot("vae", ax=ax)
    plt.title("latent space from VAE model")
    plt.show()
    s.backing.close()

# ##
# if e_():
#     data = loader_non_perturbed.__iter__().__next__()
#     data_non_perturbed = loader_non_perturbed.__iter__().__next__()
#     i0, i1 = torch.where(data[-1] == 1)
#     # perturbed data are zero, non perturbed data are ok
#     print(data[0][i0, i1])
#     print(data_non_perturbed[0][i0, i1])
#     # just a hash
#     h = np.sum(
#         np.concatenate(np.where(loader_perturbed.dataset.corrupted_entries == 1))
#     )
#     print(
#         "corrupted entries hash:",
#         h,
#     )
#
# ##
# if e_():
#     expression_vae = VAE.load_from_checkpoint(MODEL_CHECKPOINT)
#
#     debug_i = 0
#     print("merging expressions and computing embeddings... ", end="")
#     all_mu = []
#     all_expression = []
#     all_a = []
#     all_b = []
#     all_is_perturbed = []
#     all_expression_non_perturbed = []
#     for data, data_non_perturbed in tqdm(
#         zip(loader_perturbed, loader_non_perturbed),
#         desc="embedding expression",
#         total=len(loader_perturbed),
#     ):
#         expression, is_perturbed = data
#         expression_non_perturbed, _ = data_non_perturbed
#         a, b, mu, std, z = expression_vae(expression)
#         all_mu.append(mu)
#         all_expression.append(expression)
#         all_a.append(a)
#         all_b.append(b)
#         all_is_perturbed.append(is_perturbed)
#         all_expression_non_perturbed.append(expression_non_perturbed)
#         if debug_i < 5:
#             # perturbed entries
#             i0, i1 = torch.where(data[-1] == 1)
#             # non perturbed entries
#             j0, j1 = torch.where(data[-1] == 0)
#             assert torch.isclose(torch.sum(expression[i0, i1]), torch.tensor([0.0]))
#             assert torch.all(expression[j0, j1] == expression_non_perturbed[j0, j1])
#         debug_i += 1
#
#     mus = torch.cat(all_mu, dim=0)
#     expressions = torch.cat(all_expression, dim=0)
#     expressions_non_perturbed = torch.cat(all_expression_non_perturbed, dim=0)
#     a_s = torch.cat(all_a, dim=0)
#     b_s = torch.cat(all_b, dim=0)
#     are_perturbed = torch.cat(all_is_perturbed, dim=0)
#     reconstructed = expression_vae.expected_value(a_s, b_s)

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
        original=expression_val_non_perturbed.X,
        corrupted_entries=val_are_perturbed.detach().cpu().numpy(),
        predictions_from_perturbed=val_perturbed_reconstructed.detach().cpu().numpy(),
        name="visium mousebrain\nexpression vae (val)",
    )
    vae_predictions = Prediction(**kwargs)

    plt.style.use("default")
    # vae_predictions.plot_reconstruction()
    vae_predictions.plot_scores(hist=True)
    vae_predictions.plot_summary()


##
# from old_code.data2 import file_path
# import pickle
#
# if m:
#     d = {"vanilla VAE": kwargs}
#     pickle.dump(d, open(file_path("ah_scores.pickle"), "wb"))
