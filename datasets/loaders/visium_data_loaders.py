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

from datasets.visium_data import get_smu_file, get_split_indices
from utils import get_execute_function, file_path, get_bimap, print_corrupted_entries_hash

e_ = get_execute_function()
# os.environ['SPATIALMUON_NOTEBOOK'] = 'datasets/loaders/visium_data_loaders.py'

plt.style.use("dark_background")

##
class CellsDataset(Dataset):
    def __init__(self, split, only_expression=False):
        self.split = split
        self.only_expression = only_expression
        self.indices = get_split_indices(self.split)
        if not self.only_expression:
            self.tiles_file = file_path("visium_mousebrain/tiles.hdf5")
            self.f5 = h5py.File(self.tiles_file, "r")[self.split]
            assert len(self.indices) == len(self.f5)
        s = get_smu_file(read_only=True)
        self.expressions = s["visium"]["processed"].X[self.indices, :]
        s.backing.close()
        self.seed = None
        self.corrupted_entries = np.zeros(
            (len(self), self.expressions.shape[1]), dtype=np.bool
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
        print_corrupted_entries_hash(self.corrupted_entries, self.split)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        expression = self.expressions[item]
        is_corrupted = self.corrupted_entries[item]
        expression = expression * np.logical_not(is_corrupted)
        if not self.only_expression:
            image = self.f5[item][...]
            return image, expression, is_corrupted
        else:
            return expression, is_corrupted



##
if e_():
    ds = CellsDataset(split="test")
    ds.perturb()
    print(ds[0])

##
def get_cells_data_loader(split, batch_size, perturb=False, only_expression=False):
    ds = CellsDataset(split, only_expression=only_expression)
    if perturb:
        ds.perturb()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=16,
    )
    return loader

if e_():
    loader = get_cells_data_loader(split='train', batch_size=128, perturb=True, only_expression=True)
    loader.__iter__().__next__()
