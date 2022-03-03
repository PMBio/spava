##
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
from torch.utils.data import DataLoader

from utils import setup_ci, file_path
import colorama
from datasets.imc_data import get_smu_file, get_split

c_, p_, t_, n_ = setup_ci(__name__)

plt.style.use("dark_background")
##
class CellsDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.filenames = get_split(self.split)

        self.tiles_file = file_path("imc_tiles.hdf5")
        self.map_left = {}
        self.map_right = {}
        self.f5 = h5py.File(self.tiles_file, 'r')
        # with h5py.File(self.tiles_file, 'r') as f5:
        i = 0
        for filename in self.filenames:
            filename = filename.replace('.tiff', '.h5smu')
            rasters = self.f5[f'{self.split}/{filename}/raster']
            masks = self.f5[f'{self.split}/{filename}/masks']
            assert len(rasters) == len(masks)
            n = len(rasters)
            for j in range(n):
                self.map_left[i] = (filename, j)
                self.map_right[(filename, j)] = i
                i += 1

        from datasets.imc_data import compute_scaling_factors
        self.scaling_factors = compute_scaling_factors()

    def __len__(self):
        assert len(self.map_left) == len(self.map_right)
        return len(self.map_left)

    def recompute_expression(self, ome, mask):
        # recompute acculated features easily from cell tiles
        x = ome.transpose(2, 0, 1) * (mask > 0)
        e = np.sum(x, axis=(1, 2))
        e /= mask.sum()
        e = np.arcsinh(e)
        e /= self.scaling_factors
        return e

    def __getitem__(self, item):
        filename, j = self.map_left[item]
        raster = self.f5[f'{self.split}/{filename}/raster'][j, ...]
        mask = self.f5[f'{self.split}/{filename}/masks'][j, ...]
        expression = self.recompute_expression(raster, mask)
        return raster, mask, expression

##
loader = DataLoader(
    CellsDataset('train'),
    batch_size=1024,
    num_workers=16,
)
##
for x in tqdm(loader):
    pass

##
if n_ or t_ or c_:
    train_ds = CellsDataset(split="train")
    raster, mask = train_ds[0]
# class CellDataset(Dataset):
#     def __init__(
#             self,
#             split,
#             features=None,
#             perturb_pixels=False,
#             perturb_pixels_seed=42,
#             perturb_masks=False,
#     ):
#         self.features = features or {
#             "expression": True,
#             "center": False,
#             "ome": True,
#             "mask": True,
#         }
#         self.split = split
#         with h5py.File(
#                 file_path("merged_filtered_centers_and_expressions.hdf5"), "r"
#         ) as f5:
#             self.expressions = f5[f"{split}/expressions"][...]
#             self.centers = f5[f"{split}/centers"][...]
#         self.f5 = h5py.File(file_path("filtered_cells_dataset.hdf5"), "r")
#         self.f5_omes = self.f5[f"{split}/omes"]
#         self.f5_masks = self.f5[f"{split}/masks"]
#         assert len(self.expressions) == len(self.f5_omes)
#         assert len(self.expressions) == len(self.f5_masks)
#         self.length = len(self.expressions)
#         # self.n_channels = self.expressions.shape[1]
#         self.perturb_pixels = perturb_pixels
#         self.perturb_pixels_seed = perturb_pixels_seed
#         self.perturb_masks = perturb_masks
#         if self.perturb_pixels:
#             state = np.random.get_state()
#             np.random.seed(self.perturb_pixels_seed)
#             self.seeds = np.random.randint(2 ** 32 - 1, size=(self.length,))
#             np.random.set_state(state)
#         self.FORCE_RECOMPUTE_EXPRESSION = True
#         self.FRACTION_OF_PIXELS_TO_MASK = 0.25
#
#     def __len__(self):
#         return self.length
#
#     def recompute_expression(self, ome, mask):
#         # recompute acculated features easily from cell tiles
#         x = ome.transpose(2, 0, 1) * (mask > 0)
#         e = np.sum(x, axis=(1, 2))
#         e /= mask.sum()
#         e = np.arcsinh(e)
#         return e
#
#     def recompute_mask(self, mask):
#         assert self.perturb_masks
#         kernel = np.ones((3, 3), np.uint8)
#         mask_dilated = cv2.dilate(mask, kernel, iterations=1)
#         return mask_dilated
#
#     def recompute_ome(self, ome, ome_index):
#         assert self.perturb_pixels
#         state = np.random.get_state()
#         np.random.seed(self.seeds[ome_index])
#         x = np.random.binomial(1, 1 - self.FRACTION_OF_PIXELS_TO_MASK, ome.shape[:2])
#         perturbed_ome = (ome.transpose(2, 0, 1) * x).transpose(1, 2, 0)
#         np.random.set_state(state)
#         return perturbed_ome
#
#     def __getitem__(self, i):
#         l = []
#         # if mask is required
#         if (
#                 self.features["mask"]
#                 or self.perturb_masks
#                 or self.FORCE_RECOMPUTE_EXPRESSION
#         ):
#             mask = self.f5_masks[f"{i}"][...]
#             if self.perturb_masks:
#                 mask = self.recompute_mask(mask)
#         # if ome is required
#         if (
#                 self.features["ome"]
#                 or self.perturb_masks
#                 or self.perturb_pixels
#                 or self.FORCE_RECOMPUTE_EXPRESSION
#         ):
#             ome = self.f5_omes[f"{i}"][...]
#             if self.perturb_pixels:
#                 ome = self.recompute_ome(ome, ome_index=i)
#
#         if self.features["expression"]:
#             if (
#                     self.perturb_pixels
#                     or self.perturb_masks
#                     or self.FORCE_RECOMPUTE_EXPRESSION
#             ):
#                 recomputed_expression = self.recompute_expression(ome, mask)
#                 l.append(recomputed_expression)
#             else:
#                 l.append(self.expressions[i])
#         if self.features["center"]:
#             l.append(self.centers[i])
#         if self.features["ome"]:
#             l.append(ome)
#         if self.features["mask"]:
#             l.append(mask)
#         return l
