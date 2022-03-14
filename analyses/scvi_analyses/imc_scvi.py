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
from datasets.imc import (
    all_processed_smu,
    get_merged_areas_per_split,
    get_smu_file,
)
from analyses.analisys_utils import (
    scanpy_compute,
    louvain_plot,
    compare_clusters,
    compute_knn,
    nearest_neighbors,
)
from datasets.loaders.imc_loaders import CellsDatasetOnlyExpression
from datasets.imc_transform_utils import IMCPrediction, Space

e_ = get_execute_function()
# os.environ["SPATIALMUON_NOTEBOOK"] = "analyses/scvi_analyses/imc_scvi.py"
# os.environ["SPATIALMUON_TEST"] = "analyses/scvi_analyses/imc_scvi.py"

if e_():
    N_EPOCHS_KL_WARMUP = 3
    N_EPOCHS = 10
    print(f"N_EPOCHS_KL_WARMUP = {N_EPOCHS_KL_WARMUP}, N_EPOCHS = {N_EPOCHS}")

##
if e_() and False:
    # proxy for the DKFZ network
    # https://stackoverflow.com/questions/34576665/setting-proxy-to-urllib-request-python3
    os.environ["HTTP_PROXY"] = "http://193.174.53.86:80"
    os.environ["HTTPS_PROXY"] = "https://193.174.53.86:80"

    # to have a look at an existing dataset
    import scvi.data

    data = scvi.data.pbmc_dataset()
    data

##
if e_():
    d_sums = {}
    d_donors = {}
    from splits import train, validation, test

    if "SPATIALMUON_TEST" not in os.environ:
        lengths = [len(train), len(validation), len(test)]
    else:
        lengths = [1, 1, 1]
    for j, split in enumerate(
        tqdm(["train", "validation", "test"], desc="splits", position=0, leave=True)
    ):
        l0 = []
        l1 = []
        for i in tqdm(
            range(lengths[j]), desc="sums and donors", position=0, leave=True
        ):
            s = get_smu_file(split=split, index=i, read_only=True)
            x = s["imc"]["sum"].X
            l0.append(x)
            l1.extend([i] * len(x))
        sums = np.concatenate(l0, axis=0)
        sums = np.round(sums)
        sums = sums.astype(int)
        donor = np.array(l1)

        d_sums[split] = sums
        d_donors[split] = donor
    f = file_path("imc/merged_data_scvi.pickle")
    pickle.dump((d_sums, d_donors), open(f, "wb"))


def get_merged_data_scvi():
    f = file_path("imc/merged_data_scvi.pickle")
    d_sums, d_donors = pickle.load(open(f, "rb"))
    return d_sums, d_donors


##
if e_():
    d_ad = {}
    d_sums, d_donors = get_merged_data_scvi()
    for split in ["train", "validation", "test"]:
        sums = d_sums[split]
        donors = d_donors[split]
        a = ad.AnnData(sums)
        s = pd.Series(donors, index=a.obs.index)
        a.obs["batch"] = s
        d_ad[split] = a
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
    f_scvi_model = file_path("imc/scvi_model.scvi")
    # TRAIN = True
    TRAIN = False
    if not os.path.isdir(f_scvi_model):
        TRAIN = True
    print(f"{colorama.Fore.MAGENTA}TRAIN = {TRAIN}{colorama.Fore.RESET}")
    if TRAIN:
        # vae = VAE(gene_dataset.nb_genes)
        # trainer = UnsupervisedTrainer(
        #     vae,
        #     gene_dataset,
        #     train_size=0.90,
        #     use_cuda=use_cuda,
        #     frequency=5,
        # )
        # []:
        # trainer.train(n_epochs=n_epochs, lr=lr)
        model = scvi.model.SCVI(a_train)

##

# the following code, as it is, doesn't work
#     logger = TensorBoardLogger(save_dir=file_path("checkpoints"), name="scvi")
#     BATCH_SIZE = 128
#     indices = np.random.choice(len(a), BATCH_SIZE * 20, replace=False)
#
#     train_loader_batch = DataLoader(
#         a.X[indices, :],
#         batch_size=BATCH_SIZE,
#         num_workers=4,
#         pin_memory=True,
#     )
#     model.train(train_size=1., logger=logger, val_dataloaders=train_loader_batch)
#     model.__dict__
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
    if "SPATIALMUON_TEST" not in os.environ:
        random_indices = reproducible_random_choice(len(a_train), 10000)
    else:
        random_indices = reproducible_random_choice(len(a_train), len(a_train) - 1)
    aa = a_train[random_indices]
    bb = b[random_indices]


##
if e_():
    scanpy_compute(aa)
    sc.pl.pca(aa, title="pca, raw data (sum)")
    louvain_plot(aa, "UMAP with Louvain clusters, raw data (sum)")

#
if e_():
    scanpy_compute(bb)
    sc.pl.pca(bb, title="pca, scvi latent")
    louvain_plot(bb, "UMAP with Louvain clusters, scvi latent")

##
if e_():
    compare_clusters(aa, bb, description='"raw data (sum)" vs "scvi latent"')
    compute_knn(aa)
    compute_knn(bb)
    nearest_neighbors(
        nn_from=aa, plot_onto=bb, title='nn from "raw data (sum)" to "scvi latent"'
    )

##
if e_():
    # note that here with are embedding without the batch information; if you want to look at batches it does not make
    # sense to use another set except to the training one, since the train/val/test split is done by patient first
    scvi.model.SCVI.setup_anndata(
        a_val,
    )
    z_val = model.get_latent_representation(a_val)
    b_val = ad.AnnData(z_val)
    if "SPATIALMUON_TEST" not in os.environ:
        random_indices_val = reproducible_random_choice(len(a_val), 10000)
    else:
        random_indices_val = reproducible_random_choice(len(a_val), len(a_val) - 1)
    aa_val = a_val[random_indices_val].copy()
    bb_val = b_val[random_indices_val].copy()

##
if e_():
    scanpy_compute(aa_val)
    scanpy_compute(bb_val)

##
if e_():
    sc.pl.pca(aa_val, title="pca, raw data (sum); validation set")
    sc.pl.umap(
        aa_val,
        color="louvain",
        title="umap with louvain, raw data (sum); valiation set",
    )
    sc.pl.pca(bb_val, title="pca, scvi latent; valiation set")
    sc.pl.umap(
        bb_val,
        color="louvain",
        title="umap with louvain, scvi latent; valiation set",
    )

##
if e_():
    merged = ad.AnnData.concatenate(
        bb, bb_val, batch_categories=["train", "validation"]
    )
    scanpy_compute(merged)
    plt.figure()
    ax = plt.gca()
    sc.pl.umap(merged, color="batch", ax=ax, show=False)
    plt.tight_layout()
    plt.show()

##
if e_():
    size_factors = model.get_latent_library_size(a_val, give_mean=False)
    size_factors = np.squeeze(size_factors, 1)

##
if e_():
    areas = get_merged_areas_per_split()["validation"]
    assert size_factors.shape == areas.shape
    assert len(size_factors.shape) == 1

##
if e_():
    from scipy.stats import pearsonr

    r, p = pearsonr(size_factors, areas)
    plt.figure()
    plt.scatter(size_factors, areas, s=0.5)
    plt.xlabel("latent size factors")
    plt.ylabel("cell area")
    plt.title(f"r: {r:0.2f} (p: {p:0.2f})")
    plt.show()

##
# imputation benchmark
def get_corrupted_entries(split: str):
    ds = CellsDatasetOnlyExpression(split=split)
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
    a_train_perturbed.X[ce_train] = 0.0
    a_val_perturbed.X[ce_val] = 0.0
    a_test_perturbed.X[ce_test] = 0.0

##
if e_() and False:
    scvi.model.SCVI.setup_anndata(a_train_perturbed)
    # TRAIN_PERTURBED = True
    TRAIN_PERTURBED = False
    if TRAIN_PERTURBED:
        # to navigate there with PyCharm and set a breakpoint on a warning (haven't done yet)
        import scvi.core.distributions

        model = scvi.model.SCVI(a_train_perturbed)
    if TRAIN_PERTURBED:
        model.train(
            train_size=1.0, n_epochs=N_EPOCHS, n_epochs_kl_warmup=N_EPOCHS_KL_WARMUP
        )
        f = file_path("imc/scvi_model_perturbed.scvi")
        if os.path.isdir(f):
            shutil.rmtree(f)
        model.save(f)
    else:
        model = scvi.model.SCVI.load(
            file_path("imc/scvi_model_perturbed.scvi"), adata=a_train_perturbed
        )
    print(model.get_elbo())

##
if e_():
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
    uu1 = a_val.X[ce_val]

    vv0 = x_val_perturbed_pred[ne_val]
    vv1 = a_val.X[ne_val]
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
    s = np.abs(uu0 - uu1)
    t = np.abs(vv0 - vv1)
    IMCPrediction.welch_t_test(s, t)
    # reading this later: "mah"
    # the printed p-value is very close to 0
    # conclusion: the score for imputed data is worse than the one from non-perturbed data; this is expected and the
    # alternative case would have been a model whose scores are both bad because it is not properly trained
##

if e_():
    kwargs = dict(
        original=a_val.X,
        corrupted_entries=ce_val,
        predictions_from_perturbed=x_val_perturbed_pred,
        space=Space.raw_sum.value,
        name="scVI",
        split="validation",
    )
    scvi_predictions = IMCPrediction(**kwargs)

    scvi_predictions.plot_reconstruction()
    # scvi_predictions.plot_scores()

##
if e_():
    p = scvi_predictions.transform_to(Space.scaled_mean)
    p.name = "scVI scaled"
    p.plot_reconstruction()
    p.plot_scores()
    p.plot_summary()

##
import dill

if e_():
    f = file_path("imc/imputation_scores")
    os.makedirs(f, exist_ok=True)

##
if e_():
    d = {"scVI": kwargs}
    dill.dump(d, open(file_path("imc/imputation_scores/scvi_scores.pickle"), "wb"))

##
if e_():
    pickle.dump(
        {"input": aa_val, "latent": bb_val},
        open(file_path("imc/imputation_scores/latent_anndata_from_scvi.pickle"), "wb"),
    )
