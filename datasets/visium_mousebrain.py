##
import shutil
import numpy as np
import math

import pandas as pd
import spatialmuon as smu

import matplotlib.pyplot as plt
import matplotlib

import os

from utils import get_execute_function, file_path, reproducible_random_choice
import colorama
import scanpy as sc
import h5py

e_ = get_execute_function()
# os.environ['SPATIALMUON_NOTEBOOK'] = 'datasets/visium_data.py'
# os.environ['SPATIALMUON_TEST'] = 'datasets/visium_data.py'

plt.style.use("dark_background")

##
def get_smu_file(read_only: bool, initialize=False):
    "/data/spatialmuon/datasets/visium_mousebrain/smu"
    RAW_FOLDER = file_path("spatialmuon/visium_mousebrain")
    PROCESSED_FOLDER = file_path("spatialmuon_processed/visium_mousebrain")

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    if "SPATIALMUON_TEST" not in os.environ:
        src_f = os.path.join(RAW_FOLDER, "visium.h5smu")
        des_f = os.path.join(PROCESSED_FOLDER, "visium.h5smu")
    else:
        src_f = os.path.join(RAW_FOLDER, "small_visium.h5smu")
        des_f = os.path.join(PROCESSED_FOLDER, "small_visium.h5smu")

    if initialize:
        shutil.copy(src_f, des_f)
        if "SPATIALMUON_TEST" not in os.environ:
            s = smu.SpatialMuData(des_f)
            name = "ST8059049"
            name_hne = name + "H&E"
            sm = smu.SpatialModality()
            sm["expression"] = s["Visium"][name]
            sm["image"] = s["Visium"][name_hne]
            del s["Visium"]
            s["visium"] = sm
            s.backing.close()
    backingmode = "r+" if not read_only else "r"
    s = smu.SpatialMuData(des_f, backingmode=backingmode)
    return s


def get_split_indices(split):
    s = get_smu_file(read_only=True)
    n = len(s["visium"]["processed"].obs)
    s.backing.close()
    ratios = [0.7, 0.15]
    ns = [math.floor(n * ratios[0]), math.ceil(n * ratios[1])]
    ns.append(n - np.sum(ns))
    train_indices = sorted(reproducible_random_choice(n, ns[0]))
    remaining = list(set(range(n)).difference(train_indices))
    validation_indices = sorted(
        np.array(remaining)[
            np.array(reproducible_random_choice(len(remaining), ns[1]))
        ].tolist()
    )
    test_indices = sorted(list(set(remaining).difference(validation_indices)))
    assert len(set(train_indices) | set(validation_indices) | set(test_indices)) == n
    indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    return indices[split]


##
if e_():
    s = get_smu_file(initialize=True, read_only=False)
    regions = s["visium"]["expression"]
    raster = s["visium"]["image"]
##
if e_():
    _, ax = plt.subplots(1)
    raster.plot(ax=ax)
    regions.plot(0, ax=ax)
    plt.show()

##
if e_():
    # biologically relevant channels
    if "SPATIALMUON_TEST" not in os.environ:
        chosen_genes = ["Olfm1", "Plp1", "Itpka"]
    else:
        chosen_genes = ["Prex2", "Atp6v1h", "Xkr4"]
    regions.plot(channels=chosen_genes)
    raster.plot()

##
if e_():
    # x = regions.X[...]
    # if isinstance(x, ad._core.sparse_dataset.SparseDataset):
    #     x = x.todense()
    regions.masks.obs["region_center_x"] = regions.masks._masks_centers[:, 0]
    regions.masks.obs["region_center_y"] = regions.masks._masks_centers[:, 1]
    adata = smu.Converter().regions_to_anndata(regions)
    adata.var.index = regions.var["channel_name"]
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)

##
if e_():
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    adata

##
if e_():
    plt.style.use("default")
    sc.pl.violin(
        adata, ["n_genes_by_counts", "total_counts"], jitter=0.4, multi_panel=True
    )
    plt.style.use("dark_background")
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")

##
if e_():
    # we need statsmodels 0.13.2 for regress_out
    # https://stackoverflow.com/questions/71106940/cannot-import-name-centered-from-scipy-signal-signaltools
    # sc.pp.regress_out(adata, ["total_counts"])
    adata_non_scaled = adata.copy()
    sc.pp.scale(adata, max_value=10)

##
# if I use this I get a segfault from protobuf for some reasons, let us make neighbors call implicitly a version of
# pca that doesn't trigger this
if False:
    sc.pp.pca(adata)
    sc.pl.pca(adata, color=chosen_genes)
    plt.style.use("default")
    sc.pl.pca_variance_ratio(adata, log=True)
    plt.style.use("dark_background")

##
if e_():
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

if e_():
    sc.pl.umap(adata, color=chosen_genes)

##
if e_():
    sc.tl.leiden(adata)

if e_():
    sc.pl.umap(adata, color=["leiden"])
    sc.pl.spatial(adata, color=["leiden"], spot_size=20)

##
if False:
    # to understand that the data has not been log(1 + x) transformed
    x = np.asarray(adata.X.todense())
    x = np.log1p(x)
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    plt.figure()
    plt.scatter(mu, std)
    plt.xlabel("mu")
    plt.ylabel("std")
    plt.show()
# regions.var

##
if e_():
    processed_regions = smu.Regions(
        X=adata.X, var=adata.var, masks=regions.masks.clone(), anchor=regions.anchor
    )
    processed_regions.var.reset_index(inplace=True)
    processed_regions.var.rename(columns={"index": "channel_name"})
    processed_regions.masks.obs["leiden"] = adata.obs["leiden"].to_numpy()
    processed_regions.masks.obs["leiden"] = processed_regions.masks.obs[
        "leiden"
    ].astype("category")
    processed_regions.masks.set_all_has_changed(new_value=True)
    processed_regions.plot(chosen_genes)
    processed_regions.masks.plot("leiden")

    if "processed" in s["visium"]:
        del s["visium"]["processed"]
    s["visium"]["processed"] = processed_regions
    s.commit_changes_on_disk()

if e_():
    non_scaled = processed_regions.clone()
    non_scaled.X = adata_non_scaled.X

    if "non_scaled" in s["visium"]:
        del s["visium"]["non_scaled"]
    s["visium"]["non_scaled"] = non_scaled
    s.commit_changes_on_disk()

##
if e_():
    s = get_smu_file(read_only=True)
    m0 = s["visium"]["expression"].masks
    m1 = s["visium"]["processed"].masks
    m0.plot()
    m1.plot()
    s.backing.close()
##
if e_():
    s = get_smu_file(read_only=True)
    processed_regions = s["visium"]["processed"]
    raster = s["visium"]["image"]
    if "SPATIALMUON_TEST" not in os.environ:
        bb = smu.BoundingBox(x0=250, x1=1000, y0=250, y1=750)
        transformed_bb = processed_regions.anchor.transform_bounding_box(bb)
    else:
        transformed_bb = None

    plt.figure()
    ax = plt.gca()
    processed_regions.masks.plot(
        fill_colors=None, outline_colors="leiden", ax=ax, bounding_box=transformed_bb
    )
    raster.plot(ax=ax, show_legend=False, bounding_box=transformed_bb)
    raster.set_lims_to_bounding_box(transformed_bb)
    plt.show()
    s.backing.close()
##
print("done")
