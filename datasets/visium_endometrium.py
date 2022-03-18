##
import shutil
import numpy as np
import math

import pandas as pd
import spatialmuon
import spatialmuon as smu

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib

import os

from utils import (
    get_execute_function,
    file_path,
    reproducible_random_choice,
    get_bimap,
    get_splits_indices,
)
import colorama
import scanpy as sc
import h5py

e_ = get_execute_function()
# os.environ['SPATIALMUON_NOTEBOOK'] = 'datasets/visium_data.py'
# os.environ['SPATIALMUON_TEST'] = 'datasets/visium_data.py'

plt.style.use("dark_background")

if "SPATIALMUON_TEST" not in os.environ:
    visium_endrometrium_samples = ["152806", "152807", "152810", "152811"]
else:
    visium_endrometrium_samples = ["152806", "152807"]


##
def get_smu_file(sample: str, read_only: bool, initialize: bool = False):

    src_f = file_path(f"spatialmuon/visium_endometrium/{sample}.h5smu")
    des_f = file_path(f"spatialmuon_processed/visium_endometrium/{sample}.h5smu")

    os.makedirs(os.path.dirname(des_f), exist_ok=True)

    if initialize:
        shutil.copy(src_f, des_f)
    backingmode = "r+" if not read_only else "r"
    s = smu.SpatialMuData(des_f, backingmode=backingmode)
    return s


if e_():
    for sample in tqdm(visium_endrometrium_samples, "copying smu files"):
        s = get_smu_file(sample, read_only=True, initialize=True)
        print(s)
        s.backing.close()


def get_all_splits_bimaps():
    names_length_map = {}
    for sample in visium_endrometrium_samples:
        s = get_smu_file(sample, read_only=True)
        n = len(s["visium"]["expression"].obs)
        s.backing.close()
        names_length_map[sample] = n
    map_left, map_right = get_bimap(names_length_map)

    n = len(map_left)
    ratios = [0.7, 0.15]
    indices = get_splits_indices(n=n, ratios=ratios)
    map_left_per_split = {}
    map_right_per_split = {}
    for split, ii in indices.items():
        ii = set(ii)
        ml = {}
        mr = {}
        j = 0
        for i, (k, v) in enumerate(map_left.items()):
            if i in ii:
                ml[j] = v
                mr[v] = j
                j += 1
        map_left_per_split[split] = ml
        map_right_per_split[split] = mr
    return map_left_per_split, map_right_per_split


def get_split_bimap(split):
    map_left_per_split, map_right_per_split = get_all_splits_bimaps()
    return map_left_per_split[split], map_right_per_split[split]


if e_():
    ml, mr = get_split_bimap("train")
    print(ml)
    print("done")

##
def preprocess(s):
    regions = s["visium"]["expression"]
    regions.masks.obs["region_center_x"] = regions.masks._masks_centers[:, 0]
    regions.masks.obs["region_center_y"] = regions.masks._masks_centers[:, 1]
    adata = smu.Converter().regions_to_anndata(regions)
    adata.var.index = regions.var["channel_name"]
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    plt.style.use("default")
    sc.pl.violin(
        adata, ["n_genes_by_counts", "total_counts"], jitter=0.4, multi_panel=True
    )
    plt.style.use("dark_background")
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")

    # we need statsmodels 0.13.2 for regress_out
    # https://stackoverflow.com/questions/71106940/cannot-import-name-centered-from-scipy-signal-signaltools
    # sc.pp.regress_out(adata, ["total_counts"])
    adata_non_scaled = adata.copy()
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # if I use this I get a segfault from protobuf for some reasons, let us make neighbors call implicitly a version of
    # pca that doesn't trigger this
    if False:
        sc.pp.pca(adata)
        sc.pl.pca(adata, color=chosen_genes)
        plt.style.use("default")
        sc.pl.pca_variance_ratio(adata, log=True)
        plt.style.use("dark_background")

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color=adata.var_names[:3])
    sc.tl.leiden(adata)

    sc.pl.umap(adata, color=["leiden"])
    sc.pl.spatial(adata, color=["leiden"], spot_size=15)

    processed_regions = smu.Regions(
        X=adata.X, var=adata.var, masks=regions.masks.clone(), anchor=regions.anchor
    )
    processed_regions.var.reset_index(inplace=True)
    processed_regions.var.rename(columns={"index": "channel_name"})
    processed_regions.masks.obs["slide_leiden"] = adata.obs["leiden"].to_numpy()
    processed_regions.masks.obs["slide_leiden"] = processed_regions.masks.obs[
        "slide_leiden"
    ].astype("category")
    processed_regions.masks.set_all_has_changed(new_value=True)
    processed_regions.plot([0, 1])
    processed_regions.masks.plot("slide_leiden")

    if "processed" in s["visium"]:
        del s["visium"]["processed"]
    s["visium"]["processed"] = processed_regions
    s.commit_changes_on_disk()

    non_scaled = processed_regions.clone()
    non_scaled.X = adata_non_scaled.X

    if "non_scaled" in s["visium"]:
        del s["visium"]["non_scaled"]
    s["visium"]["non_scaled"] = non_scaled
    s.commit_changes_on_disk()

    return s


##
if e_():
    for sample in visium_endrometrium_samples:
        s = get_smu_file(sample, read_only=False)
        preprocess(s)
        s.backing.close()

##
if e_():
    common = {}
    for sample in visium_endrometrium_samples:
        s = get_smu_file(sample, read_only=False)
        genes = set(s['visium']['processed'].var['channel_name'].to_list())
        if len(common) == 0:
            common = genes
        else:
            common.intersection_update(genes)
        s.backing.close()

    for sample in visium_endrometrium_samples:
        s = get_smu_file(sample, read_only=False)

        def filter_var(regions, channel_names):
            ii = regions.var['channel_name'].isin(channel_names)
            new_var = regions.var[ii]
            new_x = regions.X[:, np.where(ii.to_numpy())[0]]
            regions.var = new_var
            regions.X = new_x

        filter_var(s['visium']['non_scaled'], common)
        filter_var(s['visium']['processed'], common)
        s.commit_changes_on_disk()
        s.backing.close()

##
if e_():
    for sample in visium_endrometrium_samples:
        s = get_smu_file(sample, read_only=False)
        processed_regions = s["visium"]["processed"]
        # raster = s["visium"]["image"]
        raster = s["visium"]["medium_res"]
        # raster = s["visium"]["hires_image"]
        if "SPATIALMUON_TEST" not in os.environ:
            bb = smu.BoundingBox(x0=3500, x1=5000, y0=3000, y1=4800)
        else:
            bb = None

        plt.figure()
        ax = plt.gca()
        processed_regions.masks.plot(
            fill_colors=None, outline_colors="slide_leiden", ax=ax, bounding_box=bb
        )
        raster.plot(ax=ax, show_legend=False, bounding_box=bb)
        raster.set_lims_to_bounding_box(bb)
        plt.show()
        s.backing.close()
##
if e_():
    print("done")
