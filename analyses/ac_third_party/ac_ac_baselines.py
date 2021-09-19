##
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm
from analyses.essentials import merge_perturbed_cell_dataset

from analyses.essentials import *
from data2 import PerturbedCellDataset
from graphs import CellExpressionGraphOptimized

m = __name__ == "__main__"

cells_ds_train_perturbed = PerturbedCellDataset("train")
cells_ds_train_perturbed.perturb()
cells_ds_validation_perturbed = PerturbedCellDataset("validation")
cells_ds_validation_perturbed.perturb()


expressions_train, are_perturbed_train = merge_perturbed_cell_dataset(
    cells_ds_train_perturbed
)
expressions_validation, are_perturbed_validation = merge_perturbed_cell_dataset(
    cells_ds_validation_perturbed
)
##
expressions_train.shape
are_perturbed_train.shape