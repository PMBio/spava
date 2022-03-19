##

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Union, Literal

# from tqdm.notebook import tqdm
from tqdm import tqdm

from datasets.visium_endometrium import (
    get_smu_file,
    get_split_bimap,
    visium_endrometrium_samples,
)
from utils import (
    get_execute_function,
    file_path,
    print_corrupted_entries_hash,
)

e_ = get_execute_function()

plt.style.use("dark_background")

##
class CellsDataset(Dataset):
    def __init__(
        self,
        split,
        only_expression=False,
        raw_counts=False,
        tile_dim: Union[int, Literal["large"]] = 32,
    ):
        self.split = split
        self.only_expression = only_expression
        self.map_left, self.map_right = get_split_bimap(self.split)
        self.raw_counts = raw_counts
        if not self.only_expression:
            self.tiles_file = file_path(f"visium_endometrium/tiles_{tile_dim}.hdf5")
            self.f5 = h5py.File(self.tiles_file, "r")
            assert len(self.map_left) == len(self.f5[self.split])
        self.expressions = []
        for sample in visium_endrometrium_samples:
            ##
            indices = []
            for kk, vv in self.map_left.values():
                if kk == sample:
                    indices.append(vv)
            indices = np.array(indices)

            s = get_smu_file(sample=sample, read_only=True)
            if not self.raw_counts:
                e = s["visium"]["processed"].X[indices, :]
                if type(e) != np.ndarray:
                    expressions = e.todense().A
                else:
                    expressions = e
            else:
                e = s["visium"]["non_scaled"].X[indices, :]
                if type(e) != np.ndarray:
                    expressions = e.todense().A
                else:
                    expressions = e
            self.expressions.append(expressions)
            s.backing.close()
        self.expressions = np.concatenate(self.expressions, axis=0)
        assert len(self.expressions) == len(self.map_left)

        self.seed = None
        self.corrupted_entries = np.zeros(
            (len(self), self.expressions.shape[1]), dtype=np.bool
        )

    @property
    def n_image_channels(self):
        return self.f5[self.split].shape[-1]

    @property
    def n_expression_channels(self):
        return self.expressions.shape[-1]

    @property
    def tile_dim(self):
        return self.f5[self.split].shape[1]

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
        return len(self.map_left)

    def __getitem__(self, item):
        expression = self.expressions[item]
        is_corrupted = self.corrupted_entries[item]
        expression = expression * np.logical_not(is_corrupted)
        if not self.only_expression:
            image = self.f5[self.split][item][...]
            image = image.astype(np.float32)
            m = np.max(image)
            if m > 1:
                image /= m
            return image, expression, is_corrupted
        else:
            return expression, is_corrupted


##
if e_():
    ds0 = CellsDataset(split="test")
    ds0.perturb()
    print(ds0[0])
    print(f"ds0.tile_dim = {ds0.tile_dim}")
    print(
        f"ds0.n_expression_channels = {ds0.n_expression_channels}, ds0.n_image_channels = {ds0.n_image_channels}"
    )
    ds1 = CellsDataset(split="train", only_expression=True, raw_counts=True)
    print(ds1[1])

##
def get_cells_data_loader(
    split,
    batch_size,
    perturb=False,
    only_expression=False,
    tile_dim: int = 32,
    num_workers: int = 10,
):
    ds = CellsDataset(split, only_expression=only_expression, tile_dim=tile_dim)
    if perturb:
        ds.perturb()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return loader


if e_():
    loader = get_cells_data_loader(
        split="train", batch_size=128, perturb=True, only_expression=True
    )
    loader.__iter__().__next__()

##
def get_data_per_sample(
    sample: str, only_expression=False, raw_counts=False, tile_dim=32
):
    dss = [
        CellsDataset(
            split=split,
            only_expression=only_expression,
            raw_counts=raw_counts,
            tile_dim=tile_dim,
        )
        for split in ["train", "validation", "test"]
    ]
    s = get_smu_file(sample, read_only=True)
    length = len(s["visium"]["processed"].X)
    s.backing.close()
    expressions = []
    images = []
    are_perturbed = []
    for i in tqdm(range(length)):
        in_split = []
        for j, ds in enumerate(dss):
            mr = ds.map_right
            if (sample, i) in mr:
                in_split.append(j)
        assert len(in_split) == 1
        in_split = in_split[0]
        index = dss[in_split].map_right[(sample, i)]
        data = dss[in_split][index]
        if only_expression:
            expression, is_perturbed = data
        else:
            image, expression, is_perturbed = data
            images.append(image)
        expressions.append(expression)
        are_perturbed.append(is_perturbed)
    expressions = np.stack(expressions)
    are_perturbed = np.stack(are_perturbed)
    if not only_expression:
        images = np.stack(images)
        return images, expressions, are_perturbed
    else:
        return expressions, are_perturbed


if e_():
    get_data_per_sample(visium_endrometrium_samples[0])
##
