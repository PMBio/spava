##

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# from tqdm.notebook import tqdm
from tqdm import tqdm

from datasets.imc import get_smu_file, get_split
from utils import (
    get_execute_function,
    file_path,
    get_bimap,
    print_corrupted_entries_hash,
)

e_ = get_execute_function()
# os.environ['SPATIALMUON_NOTEBOOK'] = 'datasets/loaders/imc_loaders.py'

plt.style.use("dark_background")


##
class CellsDataset(Dataset):
    def __init__(self, split, tile_dim=32, pca_tiles=False):
        self.split = split
        self.pca_tiles = pca_tiles
        self.tiles_file = file_path(f"imc/tiles_{tile_dim}.hdf5")
        self.f5 = h5py.File(self.tiles_file, "r")
        if self.pca_tiles:
            self.pca_tiles_file = file_path(f"imc/pca_tiles_{tile_dim}.hdf5")
            self.f5_pca = h5py.File(self.pca_tiles_file, "r")
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

        s = get_smu_file(split="train", index=0, read_only=True)
        self.scaling_factors = s["imc"]["transformed_mean"].uns["scaling_factors"][...]
        s.backing.close()

        assert self.tile_dim == tile_dim

        self.seed = None
        self.corrupted_entries = np.zeros(
            (len(self), len(self.scaling_factors)), dtype=np.bool
        )

    @property
    def n_image_channels(self):
        if not self.pca_tiles:
            return self.f5[f"{self.split}/{self.map_left[0][0]}/raster"].shape[-1]
        else:
            return self.f5_pca[f"{self.split}/{self.map_left[0][0]}/raster"].shape[-1]

    @property
    def n_expression_channels(self):
        return self.f5[f"{self.split}/{self.map_left[0][0]}/raster"].shape[-1]

    @property
    def tile_dim(self):
        if not self.pca_tiles:
            return self.f5[f"{self.split}/{self.map_left[0][0]}/raster"].shape[1]
        else:
            return self.f5_pca[f"{self.split}/{self.map_left[0][0]}/raster"].shape[1]

    def perturb(self, seed=0):
        self.seed = seed
        from torch.distributions import Bernoulli

        dist = Bernoulli(probs=0.1)
        shape = self.corrupted_entries.shape
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.corrupted_entries = dist.sample(shape).bool().numpy()
        torch.set_rng_state(state)
        print_corrupted_entries_hash(self.corrupted_entries, self.split)

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
        raster = np.arcsinh(raster) / self.scaling_factors
        if not raster.dtype == np.float32:
            raster = raster.astype(np.float32)

        if not self.pca_tiles:
            return raster, mask, expression, is_corrupted
        else:
            pca_raster = self.f5_pca[f"{self.split}/{filename}/raster"][j, ...]
            if not pca_raster.dtype == np.float32:
                pca_raster = pca_raster.astype(np.float32)
            return pca_raster, mask, expression, is_corrupted


##
if e_():
    train_ds = CellsDataset(split="train")
    raster, mask, expression, is_corrupted = train_ds[0]
    print(train_ds[0])
    print(f"train_ds.tile_dim = {train_ds.tile_dim}")
    print(
        f"train_ds.n_expression_channels = {train_ds.n_expression_channels}, train_ds.n_image_chann"
        f"els = {train_ds.n_image_channels}"
    )

##
if e_():
    train_ds = CellsDataset(split="train", pca_tiles=True)
    raster, mask, expression, is_corrupted = train_ds[0]


##
def get_cells_data_loader(
    split, batch_size, perturb=False, num_workers=10, pca_tiles=False
):
    ds = CellsDataset(split, pca_tiles=pca_tiles)
    if perturb:
        ds.perturb()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return loader


##
if e_():
    for split in tqdm(
        ["train", "validation", "test"], desc="split", position=0, leave=True
    ):
        list_of_expression = []
        dl = get_cells_data_loader(split=split, batch_size=1024)
        for data in tqdm(dl, desc="precomputing expression", position=0, leave=True):
            _, _, expression, is_corrupted = data
            list_of_expression.append(expression)
        expressions = torch.cat(list_of_expression, dim=0)
        s = get_smu_file(split="train", index=0, read_only=True)
        n_channels = s["imc"]["transformed_mean"].uns["scaling_factors"][...]
        os.makedirs(file_path("imc/"), exist_ok=True)
        f = file_path(f"imc/imc_merged_expressions_{split}.hdf5")
        with h5py.File(f, "w") as f5:
            f5["expressions"] = expressions.numpy()

##
class CellsDatasetOnlyExpression(Dataset):
    def __init__(self, split: str):
        self.split = split
        f = file_path(f"imc/imc_merged_expressions_{split}.hdf5")
        with h5py.File(f, "r") as f5:
            self.expressions = f5["expressions"][...]
        s = get_smu_file(split="train", index=0, read_only=True)
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
        print_corrupted_entries_hash(self.corrupted_entries, self.split)

    def __getitem__(self, item):
        expression = self.expressions[item]
        is_corrupted = self.corrupted_entries[item]
        expression[is_corrupted] = 0.0
        return expression, is_corrupted


##
if e_():
    loader = get_cells_data_loader("train", 1024)
    for x in tqdm(loader, desc="iterating cells data loader"):
        pass
