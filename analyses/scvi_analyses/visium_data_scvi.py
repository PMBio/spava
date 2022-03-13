##
from __future__ import annotations

import colorama
import shutil
import scvi
import scanpy as sc
import torch

import pickle
import numpy as np
from tqdm import tqdm
import anndata as ad
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import get_execute_function, file_path, reproducible_random_choice
from datasets.visium_data import get_smu_file, get_split_indices
from analyses.imputation_score import Prediction

from analyses.analisys_utils import (
    scanpy_compute,
    louvain_plot,
    compare_clusters,
    compute_knn,
    nearest_neighbors,
)
from datasets.loaders.visium_data_loaders import CellsDataset

e_ = get_execute_function()
os.environ["SPATIALMUON_NOTEBOOK"] = "analyses/scvi_analyses/visium_data_scvi.py"
# os.environ["SPATIALMUON_TEST"] = "analyses/scvi_analyses/visium_data_scvi.py"

if e_():
    N_EPOCHS_KL_WARMUP = 3
    N_EPOCHS = 50
    print(f"N_EPOCHS_KL_WARMUP = {N_EPOCHS_KL_WARMUP}, N_EPOCHS = {N_EPOCHS}")

##
if e_():
    d_ad = {}
    s = get_smu_file(read_only=True)

    for j, split in enumerate(
        tqdm(["train", "validation", "test"], desc="splits", position=0, leave=True)
    ):
        indices = get_split_indices(split)
        x = s['visium']['non_scaled'].X[indices]
        a = ad.AnnData(x)
        d_ad[split] = a
    s.backing.close()
##
if e_():
    a_train, a_val, a_test = (
        d_ad["train"].copy(),
        d_ad["validation"].copy(),
        d_ad["test"].copy(),
    )
##
if e_():
    scvi.model.SCVI.setup_anndata(
        a_train,
        # this is probably meaningless (if not even penalizing) for unseen data as the batches are different
        # categorical_covariate_keys=["batch"],
    )

##
if e_():
    f_scvi_model = file_path("visium/scvi_model.scvi")
    # TRAIN = True
    TRAIN = False
    if not os.path.isdir(f_scvi_model):
        TRAIN = True
    print(f'{colorama.Fore.MAGENTA}TRAIN = {TRAIN}{colorama.Fore.RESET}')
    if TRAIN:
        model = scvi.model.SCVI(a_train)

##
if e_():
    if TRAIN:
        model.train(
            train_size=1.0,
            max_epochs=N_EPOCHS,  ##, n_epochs_kl_warmup=N_EPOCHS_KL_WARMUP
        )
        if os.path.isdir(f_scvi_model):
            shutil.rmtree(f_scvi_model)
        model.save(f_scvi_model)
    else:
        model = scvi.model.SCVI.load(f_scvi_model, adata=a_train)
    print(model.get_elbo())
#
##
if e_():
    z = model.get_latent_representation()
    assert len(z) == len(a_train)
    b = ad.AnnData(z)
    aa = a_train
    bb = b

# ##
# if e_():
#     scanpy_compute(aa)
#     sc.pl.pca(aa, title="pca, raw data")
#     louvain_plot(aa, "UMAP with Louvain clusters, raw data")
#
# ##
# if e_():
#     scanpy_compute(bb)
#     sc.pl.pca(bb, title="pca, scvi latent")
#     louvain_plot(bb, "UMAP with Louvain clusters, scvi latent")
#
# ##
# if e_():
#     compare_clusters(aa, bb, description='"raw data" vs "scvi latent"')
#     compute_knn(aa)
#     compute_knn(bb)
#     nearest_neighbors(
#         nn_from=aa, plot_onto=bb, title='nn from "raw data" to "scvi latent"'
#     )

##
if e_():
    # note that here with are embedding without the batch information; if you want to look at batches it does not make
    # sense to use another set except to the training one, since the train/val/test split is done by patient first
    scvi.model.SCVI.setup_anndata(
        a_val,
    )
    z_val = model.get_latent_representation(a_val)
    b_val = ad.AnnData(z_val)
    aa_val = a_val.copy()
    bb_val = b_val.copy()

# ##
# if e_():
#     scanpy_compute(aa_val)
#     scanpy_compute(bb_val)
#
# ##
# if e_():
#     sc.pl.pca(aa_val, title="pca, raw data; validation set")
#     sc.pl.umap(
#         aa_val,
#         color="louvain",
#         title="umap with louvain, raw data; valiation set",
#     )
#     sc.pl.pca(bb_val, title="pca, scvi latent; valiation set")
#     sc.pl.umap(
#         bb_val,
#         color="louvain",
#         title="umap with louvain, scvi latent; valiation set",
#     )
#
# ##
# if e_():
#     merged = ad.AnnData.concatenate(
#         bb, bb_val, batch_categories=["train", "validation"]
#     )
#     scanpy_compute(merged)
#     plt.figure()
#     ax = plt.gca()
#     sc.pl.umap(merged, color="batch", ax=ax, show=False)
#     plt.tight_layout()
#     plt.show()

if e_():
    scvi.model.SCVI.setup_anndata(
        a_test,
    )
    z_test = model.get_latent_representation(a_test)
    b_test = ad.AnnData(z_test)
    bb_test = b_test
    merged = ad.AnnData.concatenate(bb, bb_val, bb_test, batch_categories=['train', 'validation', 'test'])
    scanpy_compute(merged)
    lou = merged.obs['louvain']
    train_indices = get_split_indices('train')
    val_indices = get_split_indices('validation')
    test_indices = get_split_indices('test')
    louvain_train = lou.iloc[:len(train_indices)]
    louvain_val = lou.iloc[len(train_indices):len(lou) - len(test_indices)]
    louvain_test = lou.iloc[len(lou) - len(test_indices): ]
    categories = lou.cat.categories.tolist() + ['Nein']
    assert len(louvain_train) + len(louvain_val) + len(louvain_test) == len(lou)
    assert all([s.endswith('-train') for s in louvain_train.index.tolist()])
    assert all([s.endswith('-validation') for s in louvain_val.index.tolist()])
    assert all([s.endswith('-test') for s in louvain_test.index.tolist()])
    ordered_lou = pd.Categorical(['Nein'] * len(lou), categories=categories)
    ordered_lou[train_indices] = louvain_train.to_numpy()
    ordered_lou[val_indices] = louvain_val.to_numpy()
    ordered_lou[test_indices] = louvain_test.to_numpy()
    assert ordered_lou.value_counts()['Nein'] == 0
    ordered_lou.remove_categories('Nein', inplace=True)
    lou_for_smu = ordered_lou.astype('category')
    s = get_smu_file(read_only=False)
    s['visium']['processed'].obs['scvi'] = lou_for_smu
    s.commit_changes_on_disk()

    print('ooo')

##
if e_():
    size_factors = model.get_latent_library_size(a_val, give_mean=False)
    size_factors = np.squeeze(size_factors, 1)

##
if e_():
    plt.figure()
    plt.hist(size_factors)
    plt.xlabel("latent size factors")
    plt.ylabel("count")
    plt.title(f"distribution of latent size factors")
    plt.show()

##
# imputation benchmark
def get_corrupted_entries(split: str):
    ds = CellsDataset(split=split)
    ds.perturb()
    corrupted_entries = ds.corrupted_entries
    return corrupted_entries


if e_():
    ce_train = get_corrupted_entries("train")
    ce_val = get_corrupted_entries("validation")
    ce_test = get_corrupted_entries("test")

##
if e_():
    a_train_perturbed, a_val_perturbed, a_test_perturbed = (
        d_ad["train"].copy(),
        d_ad["validation"].copy(),
        d_ad["test"].copy(),
    )
    a_train_perturbed.X[ce_train] = 0.
    a_val_perturbed.X[ce_val] = 0.
    a_test_perturbed.X[ce_test] = 0.

##
if e_():
    scvi.model.SCVI.setup_anndata(a_val_perturbed)
    p = model.get_likelihood_parameters(a_val_perturbed)
    from scvi.distributions import ZeroInflatedNegativeBinomial

    x_val_perturbed_pred = ZeroInflatedNegativeBinomial(
        mu=torch.tensor(p["mean"]),
        theta=torch.tensor(p["dispersions"]),
        zi_logits=torch.tensor(p["dropout"]),
    ).mean.numpy()

##
if e_():
    # ne: normal entries
    ne_train = np.logical_not(ce_train)
    ne_val = np.logical_not(ce_val)

    uu0 = x_val_perturbed_pred[ce_val]
    uu1 = a_val.X[ce_val].A1

    vv0 = x_val_perturbed_pred[ne_val]
    vv1 = a_val.X[ne_val].A1

##
if e_():
    # the two subplots should show a similar distribution
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(np.abs(uu0 - uu1))
    m = np.mean(np.abs(uu0 - uu1))
    plt.title(f"scores for imputed entries\nmean: {m:0.2f}")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.hist(np.abs(vv0 - vv1))
    m = np.mean(np.abs(vv0 - vv1))
    plt.title(f"control: normal entries\nmean: {m:0.2f}")
    plt.yscale("log")

    fig.suptitle("abs(original vs predicted)")
    plt.tight_layout()
    plt.show()

##
if e_():
    ss = np.abs(uu0 - uu1)
    tt = np.abs(vv0 - vv1)
    Prediction.welch_t_test(ss, tt)
    # for interpretation of the p-value see imc_data_scvi.py

##
if e_():
    s = get_smu_file(read_only=True)
    before = s['visium']['non_scaled'].X[...].todense().A
    after = s['visium']['processed'].X[...]
    mean = np.mean(before, axis=0)
    std = np.std(before, axis=0)
    scaled_back = np.round(after * std + mean)
    np.sum(np.logical_not(scaled_back == before))
    np.prod(scaled_back.shape)
    np.abs(scaled_back - before).max()
    # the error is too big, let's scale in the other direction. It is because of max_value being not None!

    std[std == 0.] = 1.
    manually_scaled = (before - mean) / std
    max_value = 10
    manually_scaled[manually_scaled > max_value] = max_value
    assert np.abs(manually_scaled - after).max() < 0.002
    # not super small but ok

    def scale(x):
        x = (x - mean) / std
        x[x > max_value] = max_value
        return x
    s.backing.close()

##
if e_():
    kwargs = dict(
        original=a_val.X.A,
        corrupted_entries=ce_val,
        predictions_from_perturbed=x_val_perturbed_pred,
        name="scVI (sum)",
    )
    scvi_predictions = Prediction(**kwargs)
    scvi_predictions.plot_scores(hist=True)
    scvi_predictions.plot_summary()

##
if e_():
    scvi_predictions_scaled = Prediction(
        original=scale(a_val.X.A),
        corrupted_entries=ce_val,
        predictions_from_perturbed=scale(x_val_perturbed_pred),
        name='scVI (processed)'
    )
    scvi_predictions_scaled.plot_scores(hist=True)
    scvi_predictions_scaled.plot_summary()

##
import dill

if e_():
    f = file_path('visium/imputation_scores')
    os.makedirs(f, exist_ok=True)

##
if e_():
    d = {"scVI": kwargs}
    dill.dump(d, open(file_path("visium/imputation_scores/scvi_scores.pickle"), "wb"))

##
