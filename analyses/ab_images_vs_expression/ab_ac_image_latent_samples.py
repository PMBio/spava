##
import matplotlib.cm
import numpy as np
import time
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import skimage.measure

from models.ag_conv_vae_lightning import PerturbedRGBCells
from models.ah_expression_vaes_lightning import PerturbedCellDataset

ds = PerturbedRGBCells(split="validation")

cells_ds = PerturbedCellDataset(split="validation")
if False:
    ds.perturb()
    cells_ds.perturb()

assert torch.all(ds.corrupted_entries == cells_ds.corrupted_entries)

##

models = {
    "resnet_vae": "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_7/checkpoints"
                  "/epoch=3-step=1610.ckpt",
    "resnet_vae_perturbed": "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_12"
                            "/checkpoints/last.ckpt",
    "resnet_vae_perturbed_long": "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_14/checkpoints/last.ckpt",
    "resnet_vae_last_channel": "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_20"
                               "/checkpoints/last.ckpt",
}

rgb_ds = ds.rgb_cells
from models.ag_conv_vae_lightning import VAE as ResNetVAE

the_model = "resnet_vae"
# the_model = 'resnet_vae_last_channel'
resnet_vae = ResNetVAE.load_from_checkpoint(models[the_model])
loader = DataLoader(rgb_ds, batch_size=16, num_workers=8, pin_memory=True)
data = loader.__iter__().__next__()

##
if False:
    start = time.time()
    list_of_z = []
    with torch.no_grad():
        for data in tqdm(loader, desc="embedding the whole validation set"):
            data = [d.to(resnet_vae.device) for d in data]
            z = [zz.cpu() for zz in resnet_vae(*data)]
            list_of_z.append(z)
    print(f"forwarning the data to the resnets: {time.time() - start}")

    torch.cuda.empty_cache()

##
from data2 import file_path

f = file_path("image_features.npy")
if False:
    mus = torch.cat([zz[2] for zz in list_of_z], dim=0).numpy()
    np.save(f, mus)
mus = np.load(f)
##
import scanpy as sc
import anndata as ad

a = ad.AnnData(mus)
sc.tl.pca(a)
sc.pl.pca(a)
##
random_indices = np.random.choice(len(a), 10000, replace=False)
b = a[random_indices]
##
print("computing umap... ", end="")
sc.pp.neighbors(b)
sc.tl.umap(b)
sc.tl.louvain(b)
print("done")
##
plt.figure()
l = b.obs["louvain"].tolist()
colors = list(map(int, l))
plt.scatter(
    b.obsm["X_umap"][:, 0],
    b.obsm["X_umap"][:, 1],
    s=1,
    c=colors,
    cmap=matplotlib.cm.tab20,
)
# plt.xlim([10, 20])
# plt.ylim([0, 10])
plt.show()
##
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig, ax = plt.subplots(figsize=(24, 14))
u = b.obsm["X_umap"]
for j, i in enumerate(tqdm(random_indices[:500], desc="scatterplot with images")):
    ome, mask, _ = ds[i]
    ome = ome.numpy()
    mask = torch.squeeze(mask, 0).numpy()
    im = OffsetImage(mask, zoom=0.7)
    ab = AnnotationBbox(im, u[j], xycoords="data", frameon=False)
    ax.add_artist(ab)
ax.set(xlim=(min(u[:, 0]), max(u[:, 0])), ylim=(min(u[:, 1]), max(u[:, 1])))
plt.tight_layout()
plt.show()
##
if False:
    # n_channels = ds[42][0].shape[0]
    # channels = list(range(n_channels))
    channels = [10, 34, 35, 38]
    #     ax.autoscale()
    for c in tqdm(channels, desc='channels'):
        fig, ax = plt.subplots(figsize=(24, 14))
        ax.set(title=f'ch {c}')
        u = b.obsm["X_umap"]
        for j, i in enumerate(
                tqdm(random_indices[:2000], desc="scatterplot with images", leave=False)):
            ome, _, _ = ds[i]
            ome = ome[c, :, :].numpy()
            # mask = torch.squeeze(mask, 0).numpy()
            im = OffsetImage(ome, zoom=0.7)
            ab = AnnotationBbox(im, u[j], xycoords="data", frameon=False)
            ax.add_artist(ab)
        ax.set(xlim=(min(u[:, 0]), max(u[:, 0])), ylim=(min(u[:, 1]), max(u[:, 1])))
        plt.tight_layout()
        plt.show()
##
# finding similar cells
all_expressions = []
for expression, _ in tqdm(cells_ds):
    all_expressions.append(expression.view(1, -1))
expressions = torch.cat(all_expressions, dim=0).numpy()
##
aa = ad.AnnData(expressions)
sc.tl.pca(aa)
sc.pl.pca(aa)
##
bb = aa[random_indices]
##
print("computing umap... ", end="")
sc.pp.neighbors(bb)
sc.tl.umap(bb)
sc.tl.louvain(bb)
print("done")
##
plt.figure()
l = bb.obs["louvain"].tolist()
colors = list(map(int, l))
plt.scatter(
    b.obsm["X_umap"][:, 0],
    b.obsm["X_umap"][:, 1],
    s=1,
    c=colors,
    cmap=matplotlib.cm.tab20,
)
plt.show()
##
plt.figure()
l = bb.obs["louvain"].tolist()
colors = list(map(int, l))
plt.scatter(
    bb.obsm["X_umap"][:, 0],
    bb.obsm["X_umap"][:, 1],
    s=1,
    c=colors,
    cmap=matplotlib.cm.tab20,
)
plt.show()
##
from sklearn.neighbors import NearestNeighbors
import numpy as np

random_expressions = expressions[random_indices]
nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(random_expressions)
distances, indices = nbrs.kneighbors(random_expressions)
indices
##
# plot barplots of expression for nearest neighbors, of subsampled cells
plt.style.use('dark_background')
some_cells = range(5)
rows = len(some_cells)
cols = len(indices[42])
axes = plt.subplots(rows, cols, figsize=(30, 20))[1].flatten()
k = 0
for cell in tqdm(some_cells):
    for i in indices[cell]:
        ax = axes[k]
        e = random_expressions[i]
        ax.bar(np.arange(len(e)), e)
        ax.set(title=f'cell {i}, nn of {cell}')
        k += 1
plt.tight_layout()
plt.show()
##
# plot the nearest neighbors on the umaps
for bbb in [b, bb]:
    axes = plt.subplots(1, len(some_cells), figsize=(15, 5))[1].flatten()
    k = 0
    for cell in tqdm(some_cells):
        nn = indices[cell]
        ax = axes[k]
        k += 1
        ax.scatter(
            bbb.obsm["X_umap"][:, 0],
            bbb.obsm["X_umap"][:, 1],
            s=1,
            color=(0.3, 0.3, 0.3)
        )
        ax.scatter(
            bbb.obsm["X_umap"][nn, 0],
            bbb.obsm["X_umap"][nn, 1],
            s=1,
            color='w'
        )
    plt.tight_layout()
    plt.show()
plt.style.use('default')
## study deviation of nearest neighbors from the selected cells
n_channels = ds[42][0].shape[0]
ab_for_cell = dict()
for cell in some_cells:
    list_of_squared_distances = []
    list_of_mse = []
    expression = random_expressions[cell]
    a = np.square(random_expressions - expression)
    b = np.sqrt(np.sum(a, axis=1))
    ab_for_cell[cell] = (a, b)
##
for cell, (a, b) in ab_for_cell.items():
    plt.figure()
    plt.title(f'histogram of mse from cell {cell}')
    plt.hist(b)
    plt.show()


##
# plot images for nearest neighbors, of one of the subsampled cells and for selected channels
def aaa(idx0, idx1=None):
    if idx1 is None:
        idx1 = idx0
    cell = some_cells[idx0]
    expression = random_expressions[idx1]
    selected_channels = [0, 37, 38, 3, 4, 5, 10, 35]
    axes = plt.subplots(len(selected_channels), len(indices[42]), figsize=(30, 20))[1].flatten()
    k = 0
    for channel in tqdm(selected_channels):
        for i in indices[cell]:
            ax = axes[k]
            ome, mask, _ = ds[random_indices[i]]
            im = ome[channel, :, :].numpy()
            ax.imshow(im)
            other_expression, _ = cells_ds[random_indices[i]]
            other_expression = other_expression.numpy()

            def to_simplex(x, left, right):
                return (x - left) / (right - left)

            a_for_cell, b_for_cell = ab_for_cell[idx1]
            a_for_cell = a_for_cell[:, channel]
            aaaa = np.square(other_expression - expression)
            bbbb = np.sqrt(np.sum(aaaa))
            aaaa = aaaa[channel]
            a_score = to_simplex(aaaa, np.min(a_for_cell), np.max(a_for_cell))
            b_score = to_simplex(bbbb, np.min(b_for_cell), np.max(b_for_cell))

            ax.set(title=f'{a_score:0.3f}, {b_score:0.3f}')
            k += 1
            # plot contour of mask
            numpy_mask = mask.squeeze(0).numpy()
            contours = skimage.measure.find_contours(numpy_mask, 0.4)
            # ax.imshow(numpy_mask, alpha=0.4, cmap=matplotlib.cm.gray)
            for contour in contours:
                orange = list(map(lambda x: x / 255, (255, 165, 0)))
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=orange)
    plt.tight_layout()
    plt.show()


aaa(0)
aaa(0, 1)
aaa(1)
aaa(2)
##
