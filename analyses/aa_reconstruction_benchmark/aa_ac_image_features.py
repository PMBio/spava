##
import torch

from models.ag_conv_vae_lightning import RGBCells
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models.ag_conv_vae_lightning import PerturbedRGBCells
from models.ah_expression_vaes_lightning import PerturbedCellDataset

ds = PerturbedRGBCells(split='train')
ds.perturb()

cells_ds = PerturbedCellDataset(split='train')
cells_ds.perturb()

assert torch.all(ds.corrupted_entries == cells_ds.corrupted_entries)
##
print(ds.corrupted_entries.shape)
print(cells_ds.corrupted_entries.shape)
print(ds.corrupted_entries[0, :])
print(cells_ds.corrupted_entries[0, :])
##
i = 10
c = ds.corrupted_entries[i, :]
print(c)
j = torch.nonzero(c).flatten()[0].item()
##
x = ds[i][0][j]
x_original = ds.rgb_cells[i][0][j]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(x)
plt.subplot(1, 2, 2)
plt.imshow(x_original)
plt.show()

##

