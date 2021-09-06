##
import numpy as np
import time
import torch
from tqdm import tqdm
import matplotlib.cm

from data2 import RGBCells, PerturbedRGBCells, PerturbedCellDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ds = PerturbedRGBCells(split='validation')

cells_ds = PerturbedCellDataset(split='validation')

PERTURB = False

if PERTURB:
    ds.perturb()
    cells_ds.perturb()
assert torch.all(ds.corrupted_entries == cells_ds.corrupted_entries)

##
# load the correct expression model (51)
expression_model = '/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/expression_vae/version_57' \
                   '/checkpoints/last.ckpt'
# cluster
# plot clusters
# load the correct image model
# cluster
# plot clusters
# also plot samples from the latent space
# are the clusters matching?
from models.ah_expression_vaes_lightning import VAE as ExpressionVAE, get_loaders
expression_vae = ExpressionVAE.load_from_checkpoint(expression_model)
train_loader, val_loader, train_loader_batch = get_loaders(perturb=PERTURB)
##
all_mu = []
all_expression = []
# for data in train_loader:
# data = train_loader.__iter__().__next__()
for data in tqdm(train_loader, desc='embedding expression'):
    expression, _, is_perturbed = data
    a, b, mu, std, z = expression_vae(expression)
    all_mu.append(mu)
    all_expression.append(expression)
mus = torch.cat(all_mu, dim=0)
expressions = torch.cat(all_expression, dim=0)
##
import scanpy as sc
import anndata as ad

a = ad.AnnData(mus.detach().numpy())
sc.tl.pca(a)
sc.pl.pca(a)
##
i = np.random.choice(len(a), 10000, replace=False)
b = a[i]
c = expressions[i]
##
print('computing umap... ', end='')
sc.pp.neighbors(b)
sc.tl.umap(b)
sc.tl.louvain(b)
print('done')
##
plt.figure()
l = b.obs['louvain'].tolist()
colors = list(map(int, l))
plt.scatter(b.obsm['X_umap'][:, 0], b.obsm['X_umap'][:, 1], s=1, c=colors, cmap=matplotlib.cm.tab20)
# plt.xlim([10, 20])
# plt.ylim([0, 10])
plt.show()
##
# # many barplots for visually compare expression signatures among the louvains groups. Nothing interesting out of this
# min_cell_per_label = np.min(np.unique(colors, return_counts=True)[1])
# min_cell_per_label = min(9, min_cell_per_label)
# min_cell_per_label
# n_classes = len(np.unique(colors))
# axes = plt.subplots(min_cell_per_label, n_classes, figsize=(10, 10))[1].T.flatten()
# for j0, label in enumerate(tqdm(np.unique(colors))):
#     ii = np.where(np.array(colors) == label)
#     cells = ii[0]
#     u = np.random.choice(cells, min_cell_per_label)
#     v = expressions[u]
#     for j1 in range(min_cell_per_label):
#         k = j0 * min_cell_per_label + j1
#         ax = axes[k]
#         y = v[j1].numpy()
#         x = np.arange(len(y))
#         ax.bar(x, y)
# plt.show()
##
# selected channels
# d = {0: 'subcellular', 37: 'subcellular', 38: 'subcellular',
#      3: 'boundary', 4: 'boundary', 5: 'boundary',
#      10: 'both', 35: 'both'}
