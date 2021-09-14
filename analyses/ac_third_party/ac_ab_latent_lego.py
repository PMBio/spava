##
import sys
import shutil
import scanpy as sc

from data2 import SumFilteredDataset, file_path
import numpy as np
from tqdm import tqdm
import anndata as ad
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import memory, reproducible_random_choice
import latent

# COMPLETE_RUN = True
COMPLETE_RUN = False
N_EPOCHS_KL_WARMUP = 3
N_EPOCHS = 10

m = __name__ == "__main__"

##
if m:
    ds = SumFilteredDataset("train")

    @memory.cache
    def f_xmqowidjoai(ds):
        l0 = []
        l1 = []
        for i, x in enumerate(tqdm(ds, "merging")):
            l0.append(x)
            l1.extend([i] * len(x))
        return l0, l1

    l0, l1 = f_xmqowidjoai(ds)
    x_raw_train = np.concatenate(l0, axis=0)
    x_raw_train = np.round(x_raw_train)
    x_raw_train = x_raw_train.astype(np.int)
    # donor = np.array(l1)

    ds = SumFilteredDataset("validation")

    @memory.cache
    def f_dcnmoijafqwi(ds):
        l0 = []
        l1 = []
        for i, x in enumerate(tqdm(ds, "merging")):
            l0.append(x)
            l1.extend([i] * len(x))
        return l0, l1

    l0, l1 = f_dcnmoijafqwi(ds)
    x_raw_validation = np.concatenate(l0, axis=0)
    x_raw_validation = np.round(x_raw_validation)
    x_raw_validation = x_raw_validation.astype(np.int)
    # donor = np.array(l1)

##
from analyses.aa_reconstruction_benchmark.aa_ad_reconstruction import (
    transform,
    Space,
    Prediction,
)

if m:
    x_scaled_train = transform(
        x_raw_train, from_space=Space.raw_sum, to_space=Space.scaled_mean, split="train"
    )
    x_scaled_validation = transform(
        x_raw_validation,
        from_space=Space.raw_sum,
        to_space=Space.scaled_mean,
        split="validation",
    )
##
# ---------- training with Gaussian likelihood on scaled mean ----------
from latent.models import Autoencoder

ae = Autoencoder(latent_dim=10, x_dim=x_scaled_train.shape[1], activation="relu")
ae.compile(loss="mse", optimizer="adam")
ae.fit(x_scaled_train, epochs=10, batch_size=1024)

##
x_scaled_train_latent = ae.transform(x_scaled_train)
x_scaled_validation_latent = ae.transform(x_scaled_validation)

##
from analyses.ac_third_party.ac_aa_scvi import scanpy_compute, get_corrupted_entries

ii_train = reproducible_random_choice(len(x_scaled_train_latent), 10000)
ii_validation = reproducible_random_choice(len(x_scaled_validation_latent), 10000)

a_scaled_train_latent = ad.AnnData(x_scaled_train[ii_train])
a_scaled_validation_latent = ad.AnnData(x_scaled_validation[ii_validation])

scanpy_compute(a_scaled_train_latent)
scanpy_compute(a_scaled_validation_latent)

##
from analyses.ab_images_vs_expression.ab_aa_expression_latent_samples import (
    louvain_plot,
)

louvain_plot(a_scaled_train_latent, title="scaled train latent")
louvain_plot(a_scaled_validation_latent, title="scaled validation latent")

##
a_scaled_latent = ad.AnnData.concatenate(
    a_scaled_train_latent,
    a_scaled_validation_latent,
    batch_categories=["train", "validation"],
)
scanpy_compute(a_scaled_latent)
plt.figure()
ax = plt.gca()
sc.pl.umap(a_scaled_latent, color="batch", ax=ax, show=False)
plt.tight_layout()
plt.show()

##
ce_train = get_corrupted_entries("train")
ce_validation = get_corrupted_entries("validation")

x_scaled_train_corrupted = x_scaled_train.copy()
x_scaled_train_corrupted[ce_train] = 0.0

x_scaled_validation_corrupted = x_scaled_validation.copy()
x_scaled_validation_corrupted[ce_validation] = 0.0

##
ae = Autoencoder(
    latent_dim=10, x_dim=x_scaled_train_corrupted.shape[1], activation="relu"
)
ae.compile(loss="mse", optimizer="adam")
ae.fit(x_scaled_train_corrupted, epochs=10, batch_size=1024)

x_scaled_validation_corrupted_predicted = ae.predict(x_scaled_validation_corrupted)

##
p = Prediction(
    original=x_scaled_validation,
    corrupted_entries=ce_validation,
    predictions_from_perturbed=x_scaled_validation_corrupted_predicted,
    space=Space.scaled_mean,
    name="LatentLego scaled mean",
    split="validation",
)
p.plot_reconstruction()
p.plot_scores()