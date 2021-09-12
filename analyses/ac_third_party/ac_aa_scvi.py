##
from __future__ import annotations

import sys

import scvi
import scanpy as sc
from torch.utils.data import DataLoader

from data2 import SumFilteredDataset
import numpy as np
from tqdm import tqdm
import anndata as ad
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
import os
import matplotlib.pyplot as plt

##
if False:
    # proxy for the DKFZ network
    # https://stackoverflow.com/questions/34576665/setting-proxy-to-urllib-request-python3
    os.environ["HTTP_PROXY"] = "http://193.174.53.86:80"
    os.environ["HTTPS_PROXY"] = "https://193.174.53.86:80"

##
if False:
    # to have a look at an existing dataset
    import scvi.data

    data = scvi.data.pbmc_dataset()
    data

##
ds = SumFilteredDataset("train")
l0 = []
l1 = []
for i, x in enumerate(tqdm(ds, "merging")):
    l0.append(x)
    l1.extend([i] * len(x))
raw = np.concatenate(l0, axis=0)
donor = np.array(l1)
a = ad.AnnData(raw)

##
s = pd.Series(donor, index=a.obs.index)
a.obs["batch"] = s

##
scvi.data.setup_anndata(
    a,
    # this is probably meaningless (if not even penalizing) for unseen data as the batches are different
    # categorical_covariate_keys=["batch"],
)
a

##
# TRAIN = True
TRAIN = False
if TRAIN:
    # to navigate there with PyCharm and set a breakpoint on a warning (haven't done yet)
    import scvi.core.distributions

    model = scvi.model.SCVI(a)

##
from data2 import file_path

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
if TRAIN:
    model.train(train_size=1.0, n_epochs=10)
    model.save(file_path("scvi_model.scvi"))
else:
    model = scvi.model.SCVI.load(file_path("scvi_model.scvi"), adata=a)

##
print(model.get_elbo())

##
z = model.get_latent_representation()
a.shape
z.shape
b = ad.AnnData(z)
random_indices = np.random.choice(len(a), 10000, replace=False)
aa = a[random_indices]
bb = b[random_indices]


def scanpy_compute(an: ad.AnnData):
    sc.tl.pca(an)
    print("computing neighbors... ", end="")
    sc.pp.neighbors(an)
    print("done")
    print("computing umap... ", end="")
    sc.tl.umap(an)
    print("done")
    print("computing louvain... ", end="")
    sc.tl.louvain(an)
    print("done")


##
scanpy_compute(aa)
sc.pl.pca(aa, title="pca, raw data (sum)")
sc.pl.umap(aa, color="louvain", title="umap with louvain, scvi latent (sum)")

##
scanpy_compute(bb)
sc.pl.pca(bb, title="pca, raw data (sum)")
sc.pl.umap(bb, color="louvain", title="umap with louvain, scvi latent (sum)")

##
from analyses.ab_images_vs_expression.ab_aa_expression_latent_samples import (
    compare_clusters,
    nearest_neighbors,
    compute_knn,
)

compare_clusters(aa, bb, description='"raw data (sum)" vs "scvi latent"')
compute_knn(aa)
compute_knn(bb)
nearest_neighbors(
    nn_from=aa, plot_onto=bb, title='nn from "raw data (sum)" to "scvi latent"'
)

##
ds = SumFilteredDataset("validation")
l0 = []
l1 = []
for i, x in enumerate(tqdm(ds, "merging")):
    l0.append(x)
    l1.extend([i] * len(x))
raw = np.concatenate(l0, axis=0)
donor = np.array(l1)
a_val = ad.AnnData(raw)

##
# note that here with are embedding without the batch information; if you want to look at batches it does not make
# sense to use another set except to the training one, since the train/val/test split is done by patient first
scvi.data.setup_anndata(
    a_val,
)
z_val = model.get_latent_representation(a_val)
b_val = ad.AnnData(z_val)
random_indices_val = np.random.choice(len(a_val), 10000, replace=False)
aa_val = a_val[random_indices_val]
bb_val = b_val[random_indices_val]

##
scanpy_compute(aa_val)
scanpy_compute(bb_val)

##
sc.pl.pca(aa_val, title="pca, raw data (sum); validation set")
sc.pl.umap(
    aa_val, color="louvain", title="umap with louvain, scvi latent (sum); valiation set"
)
sc.pl.pca(bb_val, title="pca, raw data (sum); valiation set")
sc.pl.umap(
    bb_val, color="louvain", title="umap with louvain, scvi latent (sum); valiation set"
)

##
merged = ad.AnnData.concatenate(bb, bb_val, batch_categories=["train", "validation"])
scanpy_compute(merged)
plt.figure()
ax = plt.gca()
sc.pl.umap(merged, color="batch", ax=ax, show=False)
plt.tight_layout()
plt.show()

##
size_factors = model.get_latent_library_size(a_val)

##
from data2 import AreaFilteredDataset

area_ds = AreaFilteredDataset("validation")

l = []
for x in tqdm(area_ds, desc="merging"):
    l.append(x)
areas = np.concatenate(l, axis=0)

##
from scipy.stats import pearsonr

print(size_factors.shape)
print(areas.shape)
r, p = pearsonr(size_factors.ravel(), areas.ravel())
plt.figure()
plt.scatter(size_factors, areas, s=0.5)
plt.xlabel("latent size factors")
plt.ylabel("cell area")
plt.title(f"r: {r:0.2f} (p: {p:0.2f})")
plt.show()

##
# imputation benchmark
from data2 import PerturbedCellDataset


def get_corrupted_entries(split: str):
    ds = PerturbedCellDataset(split)
    ds.perturb()
    corrupted_entries = ds.corrupted_entries.numpy()
    # just a hash
    h = np.sum(np.concatenate(np.where(corrupted_entries == 1)))
    print(f"corrupted entries hash ({split}):", h)
    return corrupted_entries


ce_train = get_corrupted_entries("train")
ce_val = get_corrupted_entries("validation")

##
ds = SumFilteredDataset("train")
l0 = []
for i, x in enumerate(tqdm(ds, "merging")):
    l0.append(x)
raw = np.concatenate(l0, axis=0)
raw[ce_train] = 0
a_perturbed = ad.AnnData(raw)

##
scvi.data.setup_anndata(a_perturbed)
# TRAIN_PERTURBED = True
TRAIN_PERTURBED = False
if TRAIN_PERTURBED:
    # to navigate there with PyCharm and set a breakpoint on a warning (haven't done yet)
    import scvi.core.distributions

    model = scvi.model.SCVI(a_perturbed)
if TRAIN_PERTURBED:
    model.train(train_size=1.0, n_epochs=10)
    model.save(file_path("scvi_model_perturbed.scvi"))
else:
    model = scvi.model.SCVI.load(file_path("scvi_model_perturbed.scvi"), adata=a)
print(model.get_elbo())

##
x_val_perturbed = a_val.X.copy()
x_val_perturbed[ce_val] = 0
a_val_perturbed = ad.AnnData(x_val_perturbed)

##
p = model.get_likelihood_parameters(a_val_perturbed)
from scvi.core.distributions import ZeroInflatedNegativeBinomial

x_val_perturbed_pred = ZeroInflatedNegativeBinomial(
    mu=torch.tensor(p["mean"]),
    theta=torch.tensor(p["dispersions"]),
    zi_logits=torch.tensor(p["dropout"]),
).mean.numpy()

##
# normal entries
ne_train = np.logical_not(ce_train)
ne_val = np.logical_not(ce_val)
x_val = a_val.X.copy()

uu0 = x_val_perturbed_pred[ce_val]
uu1 = x_val[ce_val]

vv0 = x_val_perturbed_pred[ne_val]
vv1 = x_val[ne_val]
##
fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(np.abs(uu0 - uu1))
m = np.median(np.abs(uu0 - uu1))
plt.title(f"scores for imputed entries\nmedian: {m:0.2f}")
plt.yscale("log")

plt.subplot(1, 2, 2)
plt.hist(np.abs(vv0 - vv1))
m = np.median(np.abs(vv0 - vv1))
plt.title(f"control: normal entries\nmedian: {m:0.2f}")
plt.yscale("log")

fig.suptitle("abs(original vs predicted)")
plt.tight_layout()
plt.show()

##
import math

s = np.abs(uu0 - uu1)
t = np.abs(vv0 - vv1)
t = np.random.choice(t, len(s), replace=False)

# Welch's t test
sem_s = s.std() / math.sqrt(len(s))
sem_t = t.std() / math.sqrt(len(t))
sed = math.sqrt(sem_s ** 2 + sem_t ** 2)
t_statistic = (s.mean() - t.mean()) / sed

# degrees of freedom from the Welch-Satterthwaite equation
delta = s - t
sem_delta = delta.std() / math.sqrt(len(delta))
df = sem_delta ** 4 / (
        (len(s) - 1) ** -1 * sem_s ** 4 + (len(t) - 1) ** -1 * sem_t ** 4
)
print(f"df = {df}")

from scipy.stats import t as t_dist

p_value = 1 - t_dist.cdf(t_statistic, df)
print(f"p_value = {p_value}")

# conclusion: the score for imputed data is worse than the one from non-perturbed data; this is expected and the
# alternative case would have been a model whose scores are both bad because it is not properly trained
##

##
# from analyses.aa_reconstruction_benchmark.aa_ad_reconstruction import Prediction

scvi_predictions = Prediction(
    original=x_val,
    corrupted_entries=ce_val,
    predictions_from_perturbed=x_val_perturbed_pred,
    space=Space.raw,
    name='scVI'
)

scvi_predictions.plot_reconstruction()

##
p = scvi_predictions.transform_to(Space.scaled)
p.plot_reconstruction()