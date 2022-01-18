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
    x_raw_train = x_raw_train.astype(int)
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
    x_raw_validation = x_raw_validation.astype(int)
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

n_channels = x_raw_train.shape[1]


##
# ---------- training with Gaussian likelihood on scaled mean ----------
def latent_lego(model_original, model_perturbed, x_train, x_validation, title: str):
    # ---------- model on non-perturbed data ----------
    # not used:
    # Fit input data using size factors for each cell
    # x_sf = np.array([1.0] * 100)
    # ae.fit([x_train, x_sf], epochs=10, batch_size=10)
    model_original.fit(x_train, epochs=10, batch_size=1024)
    x_train_latent = model_original.transform(x_train)
    x_validation_latent = model_original.transform(x_validation)

    from analyses.ac_third_party.ac_aa_scvi import scanpy_compute, get_corrupted_entries

    ii_train = reproducible_random_choice(len(x_train), 10000)
    ii_validation = reproducible_random_choice(len(x_validation), 10000)

    a_train_latent = ad.AnnData(x_train_latent[ii_train])
    a_validation_latent = ad.AnnData(x_validation_latent[ii_validation])

    scanpy_compute(a_train_latent)
    scanpy_compute(a_validation_latent)

    from analyses.ab_images_vs_expression.ab_aa_expression_latent_samples import (
        louvain_plot,
    )

    louvain_plot(a_train_latent, title=f"{title} train latent")
    louvain_plot(a_validation_latent, title=f"{title} validation latent")

    a_scaled_latent = ad.AnnData.concatenate(
        a_train_latent,
        a_validation_latent,
        batch_categories=["train", "validation"],
    )
    scanpy_compute(a_scaled_latent)
    plt.figure()
    ax = plt.gca()
    sc.pl.umap(a_scaled_latent, color="batch", ax=ax, show=False)
    plt.title(f"{title} latent")
    plt.tight_layout()
    plt.show()

    # ---------- model on perturbed data ----------
    ce_train = get_corrupted_entries("train")
    ce_validation = get_corrupted_entries("validation")

    x_train_corrupted = x_train.copy()
    x_train_corrupted[ce_train] = 0.0

    x_validation_corrupted = x_validation.copy()
    x_validation_corrupted[ce_validation] = 0.0

    model_perturbed.compile(loss="mse", optimizer="adam")
    model_perturbed.fit(x_train_corrupted, epochs=10, batch_size=1024)

    x_validation_corrupted_predicted = model_perturbed.predict(x_validation_corrupted)

    p = Prediction(
        original=x_validation,
        corrupted_entries=ce_validation,
        predictions_from_perturbed=x_validation_corrupted_predicted,
        space=Space.scaled_mean,
        name="LatentLego scaled mean",
        split="validation",
    )
    p.plot_reconstruction()
    p.plot_scores()

    p = p.transform_to(Space.raw_sum)
    p.plot_reconstruction()
    p.plot_scores()


##
from latent.models import Autoencoder


def f():
    ae = Autoencoder(latent_dim=10, x_dim=n_channels, activation="relu")
    ae.compile(loss="mse", optimizer="adam")
    return ae


# does clone_model clean also the optimizer state? let's just stay safe and create new objects
latent_lego(
    model_original=f(),
    model_perturbed=f(),
    x_train=x_scaled_train,
    x_validation=x_scaled_validation,
    title="scaled, Gaussian noise",
)

##
from latent.modules import VariationalEncoder, ZINBDecoder


def f():
    encoder = VariationalEncoder(latent_dim=10, prior="normal", kld_weight=0.01)
    decoder = ZINBDecoder(x_dim=n_channels, dispersion="gene")

    ae = Autoencoder(encoder=encoder, decoder=decoder)
    ae.compile()
    return ae


# does clone_model clean also the optimizer state? let's just stay safe
latent_lego(
    model_original=f(),
    model_perturbed=f(),
    x_train=x_scaled_train,
    x_validation=x_scaled_validation,
    title="scaled, Gaussian noise",
)
##
from latent.modules import VariationalEncoder, NegativeBinomialDecoder

# Creates a VariationalEncoder with a standard normal prior
encoder = VariationalEncoder(latent_dim=20, prior="normal", kld_weight=0.01)
# Creates a NegativeBinomialDecoder with a constant dispersion estimate
decoder = NegativeBinomialDecoder(x_dim=x_train.shape[1], dispersion="constant")

# Constructs an Autoencoder object with predefined encoder and decoder
ae = Autoencoder(encoder=encoder, decoder=decoder)
ae.compile()

# Fit input data using size factors for each cell
x_sf = np.array([1.0] * 100)
ae.fit([x_train, x_sf], epochs=10, batch_size=10)
