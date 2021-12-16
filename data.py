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

import pathlib

try:
    current_file_path = pathlib.Path(__file__).parent.absolute()

    def file_path(f):
        return os.path.join(current_file_path, "data/spatial_uzh_processed/a", f)


except NameError:
    print("setting data path manually")

    def file_path(f):
        return os.path.join("/data/l989o/data/basel_zurich/spatial_uzh_processed/a", f)


CI_TEST = "CI_TEST" in os.environ
print(f"CI_TEST = {CI_TEST}")

if __name__ == "__main__":
    if CI_TEST:
        PLOT = False
        COMPUTE = True
        DEBUG = False
    else:
        PLOT = False
        COMPUTE = False
        DEBUG = False
else:
    PLOT = False
    COMPUTE = False
    DEBUG = False


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

    def g():
        d = smu.SpatialMuData(backing=ff)
        return d

    try:
        d = g()
    except OSError as e:
        if (
            str(e)
            == "Unable to open file (file is already open for write/SWMR write (may use <h5clear "
            "file> to clear file consistency flags))"
        ):
            os.system(f"h5clear -s {ff}")
            d = g()
        else:
            raise e
    return d


#
d = get_smu_file('train', 0)
old_obs = d['imc']['masks'].masks.obs
if len(old_obs.columns) == 0:
    d['imc']['masks'].masks.update_obs_from_masks()
print(d)


##


class OmeDataset(Dataset):
    def __init__(self, split, hot_pixel_filtering=True):
        super().__init__()
        self.split = split
        self.filenames = get_split(self.split)
        self.channels_count = len(CHANNELS_TO_KEEP)
        self.hot_pixel_filtering = hot_pixel_filtering

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        filename = self.filenames[i]
        f = file_path(os.path.join("../../OMEandSingleCellMasks/ome/", filename))
        ome = skimage.io.imread(f)
        ome = np.moveaxis(ome, 0, 2)
        ome = ome[:, :, CHANNELS_TO_KEEP]

        if self.hot_pixel_filtering:
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            maxs = cv2.dilate(
                ome, kernel, iterations=1, borderType=cv2.BORDER_REFLECT101
            )
            mask = ome - maxs >= 50
            # c = ome[mask] - maxs[mask]
            # a = np.sum(c)
            # b = np.sum(ome)
            ome[mask] = maxs[mask]

        ome = torch.from_numpy(ome).float()
        return ome
