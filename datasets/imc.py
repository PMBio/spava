##
import sklearn.preprocessing
import spatialmuon as smu

import random
import math

import shutil
import copy
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import skimage
import skimage.io
import torch
import cv2
import torchvision.transforms
import os

from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import splits
import matplotlib
import os
import torch
from torch.utils.data import DataLoader  # SequentialSampler
import sys
import tempfile

import pathlib
from utils import get_execute_function, file_path
import colorama

# os.environ['SPATIALMUON_TEST'] = 'aaa'
# os.environ['SPATIALMUON_NOTEBOOK'] = 'aaa'
e_ = get_execute_function()
# matplotlib.use('module://backend_interagg')

plt.style.use("dark_background")

##
os.makedirs(file_path("imc"), exist_ok=True)


def get_split(split):
    assert split in ["train", "validation", "test"]
    ii = slice(0, 1) if "SPATIALMUON_TEST" in os.environ else slice(None)
    if split == "train":
        return splits.train[ii]
    elif split == "validation":
        return splits.validation[ii]
    else:
        return splits.test[ii]


def get_smu_file(split, index, read_only: bool, raw=False):
    if raw:
        spatialmuon_files_dir = file_path("spatialmuon/imc")
    else:
        spatialmuon_files_dir = file_path("spatialmuon_processed/imc")
    l = get_split(split)
    f = l[index]
    ff = os.path.join(spatialmuon_files_dir, f)
    ff = ff.replace(".tiff", ".h5smu")
    backingmode = "r+" if not read_only else "r"
    d = smu.SpatialMuData(backing=ff, backingmode=backingmode)
    return d


##
if e_():
    s = get_smu_file("train", 0, raw=True, read_only=False)
    old_obs = s["imc"]["masks"].masks.obs
    if len(old_obs.columns) == 0:
        s["imc"]["masks"].masks.update_obs_from_masks()
    print(s)

##
if e_():
    s["imc"]["ome"].plot(preprocessing=np.arcsinh)

##
if e_():
    s["imc"]["masks"].plot()
    s.backing.close()

## md
# channels subsetting and hot pixel filtering
##
def channels_subsetting_and_hot_pixel_filtering(s):
    PROCESSED_FOLDER = file_path("spatialmuon_processed/imc")
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    name = os.path.basename(s.backing.file.filename)
    processed_file = os.path.join(PROCESSED_FOLDER, name)
    if os.path.isfile(processed_file):
        os.unlink(processed_file)
    new_s = smu.SpatialMuData(backing=processed_file, backingmode="w")
    new_imc = smu.SpatialModality()
    new_s["imc"] = new_imc

    masks = s["imc"]["masks"].masks.X
    relabelled = skimage.measure.label(masks, connectivity=1).astype(np.uint32)
    new_masks = smu.RasterMasks(X=relabelled)
    new_imc["masks"] = smu.Regions(masks=new_masks, coordinate_unit="um")

    ## md
    ## channels subsetting
    ##
    x = s["imc"]["ome"].X[...]
    # fmt: off
    CHANNEL_NAMES = ["H3tot", "H3met", "CK5", "Fibronectin", "CK19", "CK8/18", "TWIST1", "CD68", "CK14", "SMA",
                     "Vimentin", "Myc", "HER2", "CD3", "H3phospho", "ERK1/2", "SLUG", "ER", "PR", "p53", "CD44",
                     "EpCAM", "CD45", "GATA3", "CD20", "betaCatenin", "CAIX", "Ecadherin", "Ki67", "EGFR", "S6", "Sox9",
                     "vWf_CD31", "mTOR", "CK7", "panCK", "cPARP_cCasp3", "DNA1", "DNA2", ]
    CHANNELS_TO_KEEP = [8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, ]
    # fmt: on

    if "SPATIALMUON_TEST" in os.environ:
        CHANNEL_NAMES = CHANNEL_NAMES[:10]
        CHANNELS_TO_KEEP = list(range(10))

    new_x = x[:, :, CHANNELS_TO_KEEP]

    new_var = s["imc"]["ome"].var.iloc[CHANNELS_TO_KEEP].copy()
    new_var.reset_index(inplace=True)
    new_var.rename(columns={"index": "original_index", "channel_name": "probe"})
    new_var["channel_name"] = CHANNEL_NAMES

    ## md
    ## hot pixel filtering
    ##
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    maxs = cv2.dilate(new_x, kernel, iterations=1, borderType=cv2.BORDER_REFLECT101)
    mask = new_x - maxs >= 50
    new_x[mask] = maxs[mask]

    new_x = torch.from_numpy(new_x).float()
    ##
    new_imc["ome"] = smu.Raster(X=new_x, var=new_var, coordinate_unit="um")


##
def u(raw: bool, read_only: bool):
    for split in tqdm(
        ["train", "validation", "test"], desc="split", position=0, leave=True
    ):
        for index in tqdm(
            range(len(get_split(split))), desc="slide", position=0, leave=True
        ):
            s = get_smu_file(split=split, index=index, read_only=read_only, raw=raw)
            yield s


def all_processed_smu():
    return u(raw=False, read_only=False)


def all_raw_smu():
    return u(raw=True, read_only=False)


##
if e_():
    print(
        f"{colorama.Fore.MAGENTA}channel subsetting, hot-pixel filtering, masks relabeling:{colorama.Fore.RESET}"
    )
    for s in all_raw_smu():
        channels_subsetting_and_hot_pixel_filtering(s)
        s.backing.close()

##
if e_():
    s = get_smu_file(split="train", index=0, read_only=True, raw=False)
    s["imc"]["ome"].plot(preprocessing=np.arcsinh)
    s["imc"]["ome"].plot(channels=0, preprocessing=np.arcsinh)
    s.backing.close()

## md
# accumulate features
##
if e_():
    print(f"{colorama.Fore.MAGENTA}feature accumulation:{colorama.Fore.RESET}")
    for s in all_processed_smu():
        accumulated = s["imc"]["ome"].accumulate_features(s["imc"]["masks"].masks)
        for k in ["mean", "sum"]:
            if k in s["imc"]:
                del s["imc"][k]
            s["imc"][k] = accumulated[k]
        s.backing.close()
        # for k in accumulated.keys():
        #     del s["imc"][k]
##
## md
# filter small cells
##
if e_():
    CUTOFF = 20
    areas = []
    for s in all_processed_smu():
        a = s["imc"]["mean"].masks.obs["count"]
        areas.append(a.to_numpy())
        s.backing.close()
    areas = np.concatenate(areas)
    ##
    plt.figure()
    plt.hist(areas, bins=100)
    plt.axvline(x=CUTOFF)
    plt.title("cutoff to filter cells by area")
    plt.xlabel("cell area")
    plt.ylabel("count")
    plt.show()
    ##
    plt.figure()
    plt.hist(areas, bins=1000)
    plt.xlim([0, 100])
    plt.axvline(x=CUTOFF)
    plt.title("cutoff to filter cells by area")
    plt.xlabel("cell area")
    plt.ylabel("count")
    plt.show()
    ##
    print(f"{colorama.Fore.MAGENTA}filtering cells by area{colorama.Fore.RESET}")
    for s in all_processed_smu():
        o = s["imc"]["mean"].masks.obs
        obs_to_keep = o["count"] > CUTOFF
        indices_to_keep = np.where(obs_to_keep.to_numpy())[0]
        all_indices = o.index.to_numpy()
        indices_to_discard = np.setdiff1d(all_indices, indices_to_keep)

        def new_regions_obj(regions):
            _mask = regions.masks.X[...]
            labels_to_remove = (
                regions.masks.obs["original_labels"].iloc[indices_to_discard].tolist()
            )
            for ii in labels_to_remove:
                assert np.sum(_mask == ii) <= CUTOFF
                _mask[_mask == ii] = 0
            obs = regions.masks.obs.iloc[indices_to_keep]
            # obs.reset_index()
            new_masks = smu.RasterMasks(X=_mask)
            new_masks._obs = obs
            assert regions.is_backed
            if "X" in regions.backing:
                new_x = regions.X
                new_x = new_x[...][indices_to_keep, :]
            else:
                new_x = None
            new_regions = smu.Regions(X=new_x, masks=new_masks, var=regions.var)
            return new_regions

        old_masks = s["imc"]["masks"]
        old_mean = s["imc"]["mean"]
        old_sum = s["imc"]["sum"]
        new_masks = new_regions_obj(old_masks)
        new_mean = new_regions_obj(old_mean)
        new_sum = new_regions_obj(old_sum)
        if "filtered_masks" in s["imc"]:
            del s["imc"]["filtered_masks"]
        if "filtered_mean" in s["imc"]:
            del s["imc"]["filtered_mean"]
        if "filtered_sum" in s["imc"]:
            del s["imc"]["filtered_sum"]
        s["imc"]["filtered_masks"] = new_masks
        s["imc"]["filtered_mean"] = new_mean
        s["imc"]["filtered_sum"] = new_sum
        s.backing.close()

##
if e_():
    s = get_smu_file(split="train", index=0, read_only=True)
    _, ax = plt.subplots(1, figsize=(20, 20))
    s["imc"]["masks"].masks.plot(fill_colors="red", ax=ax)
    # s['imc']['filtered_mean'].masks.plot(fill_colors='black', ax=ax)
    s["imc"]["filtered_mean"].plot(0, ax=ax)
    plt.suptitle("cells filtered because too small shown in red")
    plt.show()
    s.backing.close()

##
def compute_scaling_factors():
    all_expressions = []
    for index in tqdm(
        range(len(get_split("train"))),
        desc="computing scaling factors",
        position=0,
        leave=True,
    ):
        s = get_smu_file(split="train", index=index, read_only=True)
        e = s["imc"]["mean"].X[...]
        all_expressions.append(e)
        s.backing.close()
    expressions = np.concatenate(all_expressions, axis=0)
    transformed = np.arcsinh(expressions)
    quantile = 0.9
    scaling_factors = np.quantile(transformed, q=quantile, axis=0)
    return scaling_factors


##
if e_():
    print(f"{colorama.Fore.MAGENTA}scaling accumulated data{colorama.Fore.RESET}")
    scaling_factors = compute_scaling_factors()
    for s in all_processed_smu():
        regions = s["imc"]["mean"]
        new_x = np.arcsinh(regions.X[...]) / scaling_factors
        new_regions = smu.Regions(X=new_x, masks=regions.masks, var=regions.var)
        if "transformed_mean" in s["imc"]:
            del s["imc"]["transformed_mean"]
        s["imc"]["transformed_mean"] = new_regions
        s["imc"]["transformed_mean"].uns["scaling_factors"] = scaling_factors
        s.commit_changes_on_disk()
        s.backing.close()
##
if e_():
    s0 = get_smu_file(split="train", index=0, read_only=True)
    s1 = get_smu_file(split="validation", index=0, read_only=True)
    assert np.alltrue(
        s0["imc"]["transformed_mean"].uns["scaling_factors"][...]
        == s1["imc"]["transformed_mean"].uns["scaling_factors"][...]
    )
    s0.backing.close()
    s1.backing.close()

##
if e_():
    d = {}
    from splits import train, validation, test

    if "SPATIALMUON_TEST" not in os.environ:
        lengths = [len(train), len(validation), len(test)]
    else:
        lengths = [1, 1, 1]
    for j, split in enumerate(
        tqdm(["train", "validation", "test"], desc="splits", position=0, leave=True)
    ):
        l = []
        for i in tqdm(range(lengths[j]), desc="cell areas", position=0, leave=True):
            s = get_smu_file(split=split, index=i, read_only=True)
            x = s["imc"]["transformed_mean"].obs["count"].to_numpy()
            l.append(x)
            s.backing.close()
        areas = np.concatenate(l, axis=0)
        d[split] = areas
    f = file_path("imc/merged_areas_per_split.pickle")
    pickle.dump(d, open(f, "wb"))


def get_merged_areas_per_split():
    f = file_path("imc/merged_areas_per_split.pickle")
    d = pickle.load(open(f, "rb"))
    return d


if e_():
    areas_per_split = get_merged_areas_per_split()

##
# if True:
if e_():
    print(
        f"{colorama.Fore.MAGENTA}preprocessing images for conv nets{colorama.Fore.RESET}"
    )
    all_x = []
    for i in tqdm(range(len(get_split("train"))), desc="merging expressions"):
        s = get_smu_file("train", i, read_only=True)
        x = s["imc"]["transformed_mean"].X[...]
        all_x.append(x)
        s.backing.close()
    merged_x = np.concatenate(all_x, axis=0)
    ##
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=x.shape[1])
    pca_fit = reducer.fit(merged_x)
    ##
    pc = np.arange(reducer.n_components_) + 1
    plt.figure()
    plt.plot(pc, reducer.explained_variance_ratio_, "o-", linewidth=2)
    plt.title(f"Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.show()
    ##
    s = get_smu_file("train", 0, read_only=True)
    scaling_factors = s["imc"]["transformed_mean"].uns["scaling_factors"][...]
    s.backing.close()
    ##
    import sklearn.preprocessing

    scaler = sklearn.preprocessing.StandardScaler()
    n = 6
    d = {
        "train": get_split("train"),
        "validation": get_split("validation"),
        "test": get_split("test"),
    }
    for k, v in tqdm(d.items(), desc="split", position=0, leave=True):
        for i in tqdm(range(len(v)), desc="computing pca", position=0, leave=True):
            s = get_smu_file(k, i, read_only=False)
            ome = s["imc"]["ome"].X[...]
            ome = np.arcsinh(ome)
            ome /= scaling_factors
            reshaped = ome.reshape(-1, len(scaling_factors))
            components = reducer.transform(reshaped)[:, :n]
            if k == "train":
                scaler.partial_fit(components)
            ome_pc = components.reshape(ome.shape[0], ome.shape[1], n)
            if "ome_pc" in s["imc"]:
                del s["imc"]["ome_pc"]
            s["imc"]["ome_pc"] = smu.Raster(X=ome_pc, anchor=s["imc"]["ome"].anchor)
            s.backing.close()

    print(f"scaler.mean_ = {scaler.mean_}, scaler.var_ = {scaler.var_}")
    ##
    minimum = np.array([sys.float_info.max] * n)
    maximum = np.array([sys.float_info.min] * n)
    for k, v in tqdm(d.items(), desc="split", position=0, leave=True):
        for i in tqdm(range(len(v)), desc="standardization", position=0, leave=True):
            s = get_smu_file(k, i, read_only=False)
            x = s["imc"]["ome_pc"].X
            x = (x - scaler.mean_) / np.sqrt(scaler.var_)
            a = np.amin(x, axis=(0, 1))
            b = np.amax(x, axis=(0, 1))
            minimum = np.minimum(minimum, a)
            maximum = np.maximum(maximum, b)
            s["imc"]["ome_pc"].X = x
            s.commit_changes_on_disk()
            s.backing.close()
    ##
    assert np.all(maximum - minimum > 1e-3)
    ##
    for k, v in tqdm(d.items(), desc="split", position=0, leave=True):
        for i in tqdm(
            range(len(v)), desc="affine transform to [0, 1]", position=0, leave=True
        ):
            s = get_smu_file(k, i, read_only=False)
            x = s["imc"]["ome_pc"].X
            x = (x - minimum) / (maximum - minimum)
            s["imc"]["ome_pc"].X = x
            s.commit_changes_on_disk()
            s.backing.close()
    ##
    s = get_smu_file("train", 0, read_only=True)
    x = s["imc"]["ome_pc"].X[...]
    assert np.min(x) >= 0.0
    assert np.max(x) <= 1.0
    _, axes = plt.subplots(2, 1)
    channels = slice(0, 3)
    axes[0].imshow(x[:, :, channels])
    axes[1].hist(x[:, :, channels].flatten(), bins=50)
    plt.show()
    s.backing.close()

##

print("done")
