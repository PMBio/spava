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

from datasets.visium_mousebrain import get_smu_file, get_split_indices
from utils import (
    get_execute_function,
    file_path,
    get_bimap,
    print_corrupted_entries_hash,
)

e_ = get_execute_function()
# os.environ['SPATIALMUON_TEST'] = 'datasets/loaders/visium_mousebrain_loaders.py'

plt.style.use("dark_background")

##
class CellsDataset(Dataset):
    def __init__(
        self, split, only_expression=False, raw_counts=False, tile_dim: int = 32
    ):
        self.split = split
        self.only_expression = only_expression
        self.indices = get_split_indices(self.split)
        self.raw_counts = raw_counts
        if not self.only_expression:
            self.tiles_file = file_path(f"visium_mousebrain/tiles_{tile_dim}.hdf5")
            self.f5 = h5py.File(self.tiles_file, "r")[self.split]
            assert len(self.indices) == len(self.f5)
        s = get_smu_file(read_only=True)
        if not self.raw_counts:
            self.expressions = s["visium"]["processed"].X[self.indices, :]
        else:
            self.expressions = s["visium"]["non_scaled"].X[self.indices, :].todense().A
        s.backing.close()

        if not self.only_expression:
            assert self.tile_dim == tile_dim

        self.seed = None
        self.corrupted_entries = np.zeros(
            (len(self), self.expressions.shape[1]), dtype=np.bool
        )

    @property
    def n_image_channels(self):
        return self.f5[0].shape[-1]

    @property
    def n_expression_channels(self):
        return self.expressions.shape[-1]

    @property
    def tile_dim(self):
        return self.f5[0].shape[0]

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
        return len(self.indices)

    def __getitem__(self, item):
        expression = self.expressions[item]
        is_corrupted = self.corrupted_entries[item]
        expression = expression * np.logical_not(is_corrupted)
        if not self.only_expression:
            image = self.f5[item][...]
            image = image.astype(np.float32) / 255.0
            return image, expression, is_corrupted
        else:
            return expression, is_corrupted


##
if e_():
    ds = CellsDataset(split="test")
    ds.perturb()
    print(ds[0])
    print(f"ds.tile_dim = {ds.tile_dim}")
    print(
        f"ds.n_expression_channels = {ds.n_expression_channels}, ds.n_image_channels = {ds.n_image_channels}"
    )

##
def get_cells_data_loader(split, batch_size, perturb=False, only_expression=False, tile_dim: int = 32):
    ds = CellsDataset(split, only_expression=only_expression, tile_dim=tile_dim)
    if perturb:
        ds.perturb()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=10,
    )
    return loader


if e_():
    loader = get_cells_data_loader(
        split="train", batch_size=128, perturb=True, only_expression=True
    )
    loader.__iter__().__next__()
