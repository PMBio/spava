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
from torch_geometric.data import DataLoader as GeometricDataLoader

from utils import get_execute_function, file_path, get_bimap
import colorama
from datasets.imc_data import get_smu_file, get_split

e_ = get_execute_function()
# os.environ['SPATIALMUON_NOTEBOOK'] = 'datasets/loaders/imc_data_loaders.py'

plt.style.use("dark_background")


##
class CellsDataset(Dataset):
    def __init__(self, split):
        self.split = split

        self.tiles_file = file_path("imc_tiles.hdf5")

        self.f5 = h5py.File(self.tiles_file, "r")
        self.filenames = get_split(self.split)
        names_length_map = {}
        for filename in self.filenames:
            filename = filename.replace(".tiff", ".h5smu")
            rasters = self.f5[f"{self.split}/{filename}/raster"]
            masks = self.f5[f"{self.split}/{filename}/masks"]
            assert len(rasters) == len(masks)
            n = len(rasters)
            names_length_map[filename] = n
        self.map_left, self.map_right = get_bimap(names_length_map)

        # with h5py.File(self.tiles_file, 'r') as f5:
        i = 0
        for filename in self.filenames:
            filename = filename.replace(".tiff", ".h5smu")
            rasters = self.f5[f"{self.split}/{filename}/raster"]
            masks = self.f5[f"{self.split}/{filename}/masks"]
            assert len(rasters) == len(masks)
            n = len(rasters)
            for j in range(n):
                self.map_left[i] = (filename, j)
                self.map_right[(filename, j)] = i
                i += 1

        s = get_smu_file("train", 0, read_only=True)
        self.scaling_factors = s["imc"]["transformed_mean"].uns["scaling_factors"][...]
        s.backing.close()
        self.seed = None
        self.corrupted_entries = np.zeros(
            (len(self), len(self.scaling_factors)), dtype=np.bool
        )

    def perturb(self, seed=0):
        self.seed = seed
        from torch.distributions import Bernoulli

        dist = Bernoulli(probs=0.1)
        shape = self.corrupted_entries.shape
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.corrupted_entries = dist.sample(shape).bool().numpy()
        torch.set_rng_state(state)

    def __len__(self):
        assert len(self.map_left) == len(self.map_right)
        return len(self.map_left)

    def recompute_expression(self, ome, mask):
        # recompute acculated features easily from cell tiles
        assert len(mask.shape) == 3
        mask = mask.squeeze(2)
        assert len(mask.shape) == 2
        x = ome.transpose(2, 0, 1) * (mask > 0)
        e = np.sum(x, axis=(1, 2))
        e /= mask.sum()
        e = np.arcsinh(e)
        e /= self.scaling_factors
        return e

    def __getitem__(self, item):
        filename, j = self.map_left[item]
        raster = self.f5[f"{self.split}/{filename}/raster"][j, ...]
        is_corrupted = self.corrupted_entries[item]
        raster *= np.logical_not(is_corrupted)
        mask = self.f5[f"{self.split}/{filename}/masks"][j, ...]
        expression = self.recompute_expression(raster, mask)
        return raster, mask, expression, is_corrupted

##
if e_():
    train_ds = CellsDataset(split="train")
    raster, mask, expression, is_corrupted = train_ds[0]


##
def get_cells_data_loader(split, batch_size, perturb=False):
    ds = CellsDataset(split)
    if perturb:
        ds.perturb()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=16,
    )
    return loader


##
if e_():
    for split in tqdm(['train', 'validation', 'test'], desc='split', position=0, leave=True):
        list_of_expression = []
        dl = get_cells_data_loader(split=split, batch_size=1024)
        for data in tqdm(dl, desc='precomputing expression', position=0, leave=True):
            _, _, expression, is_corrupted = data
            list_of_expression.append(expression)
        expressions = torch.cat(list_of_expression, dim=0)
        s = get_smu_file("train", 0, read_only=True)
        n_channels = s["imc"]["transformed_mean"].uns["scaling_factors"][...]
        f = file_path(f'imc_merged_expressions_{split}.hdf5')
        with h5py.File(f, 'w') as f5:
            f5['expressions'] = expressions.numpy()

##
class CellsDatasetOnlyExpression(Dataset):
    def __init__(self, split: str):
        self.split = split
        f = file_path(f'imc_merged_expressions_{split}.hdf5')
        with h5py.File(f, 'r') as f5:
            self.expressions = f5['expressions'][...]
        s = get_smu_file("train", 0, read_only=True)
        self.scaling_factors = s["imc"]["transformed_mean"].uns["scaling_factors"][...]
        s.backing.close()
        self.seed = None
        self.corrupted_entries = np.zeros(
            (len(self), len(self.scaling_factors)), dtype=np.bool
        )

    def __len__(self):
        return len(self.expressions)

    def perturb(self, seed=0):
        self.seed = seed
        from torch.distributions import Bernoulli

        dist = Bernoulli(probs=0.1)
        shape = self.corrupted_entries.shape
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.corrupted_entries = dist.sample(shape).bool().numpy()
        torch.set_rng_state(state)

    def __getitem__(self, item):
        expression = self.expressions[item]
        is_corrupted = self.corrupted_entries[item]
        expression[is_corrupted] = 0.
        return expression, is_corrupted

##
if e_():
    loader = get_cells_data_loader("train", 1024)
    for x in tqdm(loader, desc="iterating cells data loader"):
        pass

##
def get_graphs_data_loader(split, subgraph_name, batch_size):
    from datasets.graphs.imc_data_graphs import CellGraphsDataset

    ds = CellGraphsDataset(split=split, name=subgraph_name)
    ##
    loader = GeometricDataLoader(
        ds,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True,
    )
    return loader


if e_():
    loader = get_graphs_data_loader(
        split="train", subgraph_name="knn_10_max_distance_in_units_50", batch_size=64
    )
    for x in tqdm(loader, desc="iterating graph data loader"):
        pass

##
