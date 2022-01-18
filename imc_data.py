##
import spatialmuon as smu

import random
import math

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

import pathlib
from utils import setup_ci, file_path

c_, p_, d_ = setup_ci(__name__)

plt.style.use("dark_background")

def get_split(split):
    assert split in ["train", "validation", "test"]
    if split == "train":
        return splits.train
    elif split == "validation":
        return splits.validation
    else:
        return splits.test


#
def get_smu_file(split, index):
    spatialmuon_files_dir = file_path("spatialmuon/")
    l = get_split(split)
    f = l[index]
    ff = os.path.join(spatialmuon_files_dir, f)
    ff = ff.replace(".tiff", ".h5smu")
    d = smu.SpatialMuData(backing=ff)
    return d


#
if c_:
    d = get_smu_file("train", 0)
    old_obs = d["imc"]["masks"].masks.obs
    if len(old_obs.columns) == 0:
        d["imc"]["masks"].masks.update_obs_from_masks()
    print(d)

##
if p_:
    d["imc"]["ome"].plot(preprocessing=np.arcsinh)

##
if p_:
    d["imc"]["masks"].plot()
