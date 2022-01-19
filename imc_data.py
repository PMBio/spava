##
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
import vigra
import os

# from tqdm.notebook import tqdm
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm import tqdm
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
from utils import setup_ci, file_path

c_, p_, t_, n_ = setup_ci(__name__)

plt.style.use("dark_background")

##
RAW_FOLDER = file_path('spatialmuon')
PROCESSED_FOLDER = file_path("spatialmuon_processed")


def get_split(split):
    assert split in ["train", "validation", "test"]
    if split == "train":
        return splits.train
    elif split == "validation":
        return splits.validation
    else:
        return splits.test


def get_smu_file(split, index, raw=False):
    if raw:
        spatialmuon_files_dir = RAW_FOLDER
    else:
        spatialmuon_files_dir = PROCESSED_FOLDER
    l = get_split(split)
    f = l[index]
    ff = os.path.join(spatialmuon_files_dir, f)
    ff = ff.replace(".tiff", ".h5smu")
    d = smu.SpatialMuData(backing=ff)
    return d


##
if n_ or t_ or c_ and False:
    d = get_smu_file("train", 0, raw=True)
    old_obs = d["imc"]["masks"].masks.obs
    if len(old_obs.columns) == 0:
        d["imc"]["masks"].masks.update_obs_from_masks()
    print(d)

##
if n_ or t_ or p_ and False:
    d["imc"]["ome"].plot(preprocessing=np.arcsinh)

##
if n_ or t_ or p_ and False:
    d["imc"]["masks"].plot()

## md
# channels subsetting and hot pixel filtering
##
def channels_subsetting_and_hot_pixel_filtering(s):
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    name = os.path.basename(s.backing.file.filename)
    processed_file = os.path.join(PROCESSED_FOLDER, name)
    if os.path.isfile(processed_file):
        os.unlink(processed_file)
    new_s = smu.SpatialMuData(backing=processed_file)
    new_imc = smu.SpatialModality()
    new_s["imc"] = new_imc
    new_masks = copy.copy(s["imc"]["masks"])
    new_imc["masks"] = new_masks

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
    new_x = x[:, :, CHANNELS_TO_KEEP]

    new_var = s["imc"]["ome"].var.iloc[CHANNELS_TO_KEEP]
    new_var.reset_index(inplace=True)
    new_var.rename(columns={"index": "original_index", 'channel_name': 'probe'})
    new_var['channel_name'] = CHANNEL_NAMES

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
def u(raw: bool):
    for split in tqdm(["train", 'validation', 'test'], desc="split"):
        for index in tqdm(range(len(get_split(split))), desc='slide'):
            s = get_smu_file(split, index, raw=raw)
            yield s

def all_processed_smu():
    return u(raw=False)

def all_raw_smu():
    return u(raw=True)

##
if n_ or t_ or c_ and False:
    for s in all_raw_smu():
        channels_subsetting_and_hot_pixel_filtering(s)
        if t_:
            break

##
if n_ or t_ or p_ and False:
    s = get_smu_file("train", 0, raw=False)
    s["imc"]["ome"].plot(preprocessing=np.arcsinh)
    s['imc']['ome'].plot(channels='DNA1', preprocessing=np.arcsinh)

## md
# accumulate features
##
if n_ or t_ or c_ and False:
    for s in all_processed_smu():
        accumulated = s["imc"]["ome"].accumulate_features(s["imc"]["masks"].masks)
        k = 'mean'
        if k in s["imc"]:
            del s["imc"][k]
        s["imc"][k] = accumulated[k]
        # for k in accumulated.keys():
        #     del s["imc"][k]
        if t_:
            break
##
## md
# filter small cells
##
if n_ or t_ or c_ and False:
    CUTOFF = 20
    if n_ or t_ or p_ and False:
        areas = []
        for s in all_processed_smu():
            a = s['imc']['mean'].masks.obs['count']
            areas.append(a.to_numpy())
        areas = np.concatenate(areas)
        ##
        plt.figure()
        plt.hist(areas, bins=100)
        plt.axvline(x=CUTOFF)
        plt.title('cutoff to filter cells by area')
        plt.xlabel('cell area')
        plt.ylabel('count')
        plt.show()
        ##
        plt.figure()
        plt.hist(areas, bins=1000)
        plt.xlim([0, 100])
        plt.axvline(x=CUTOFF)
        plt.title('cutoff to filter cells by area')
        plt.xlabel('cell area')
        plt.ylabel('count')
        plt.show()
    ##
    for s in all_processed_smu():
        o = s['imc']['mean'].masks.obs
        obs_to_keep = o['count'] > CUTOFF
        indices_to_keep = np.where(obs_to_keep.to_numpy())[0]
        all_indices = o.index.to_numpy()
        indices_to_discard = np.setdiff1d(all_indices, indices_to_keep)

        def new_regions_obj(regions):
            _mask = regions.masks._mask[...]
            labels_to_remove = regions.masks.obs['original_labels'].iloc[indices_to_discard].tolist()
            for ii in labels_to_remove:
                assert np.sum(_mask == ii) <= CUTOFF
                _mask[_mask == ii] = 0
            obs = regions.masks.obs.iloc[indices_to_keep]
            # obs.reset_index()
            new_masks = smu.RasterMasks(mask=_mask)
            new_masks._obs = obs
            assert regions.isbacked
            if 'X' in regions.backing:
                new_x = regions.X
                new_x = new_x[indices_to_keep, :]
            else:
                new_x = None
            new_regions = smu.Regions(X=new_x, masks=new_masks, var=regions.var)
            return new_regions
        old_masks = s['imc']['masks']
        old_mean = s['imc']['mean']
        new_masks = new_regions_obj(old_masks)
        new_mean = new_regions_obj(old_mean)
        if 'filtered_masks' in s['imc']:
            del s['imc']['filtered_masks']
        if 'filtered_mean' in s['imc']:
            del s['imc']['filtered_mean']
        s['imc']['filtered_masks'] = new_masks
        s['imc']['filtered_mean'] = new_mean
        if t_:
            break

##
if n_ or t_ or p_ and False:
    s = get_smu_file('train', 2)
    _, ax = plt.subplots(1, figsize=(20, 20))
    s['imc']['masks'].masks.plot(fill_colors='red', ax=ax)
    # s['imc']['filtered_mean'].masks.plot(fill_colors='black', ax=ax)
    s['imc']['filtered_mean'].plot(0, ax=ax)
    plt.suptitle('cells filtered because too small shown in red')
    plt.show()

##
