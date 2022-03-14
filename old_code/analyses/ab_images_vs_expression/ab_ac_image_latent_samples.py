##
import matplotlib.cm
import numpy as np
import time
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import skimage.measure
from old_code.data2 import (
    PerturbedRGBCells,
    PerturbedCellDataset,
    file_path,
    CHANNEL_NAMES,
)
import scanpy as sc
import anndata as ad
from utils import reproducible_random_choice
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

m = __name__ == "__main__"

ds = PerturbedRGBCells(split="validation")

cells_ds = PerturbedCellDataset(split="validation")
if False:
    ds.perturb()
    cells_ds.perturb()

assert torch.all(ds.corrupted_entries == cells_ds.corrupted_entries)

##

models = {
    # "resnet_vae": "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_7/old_checkpoints"
    #               "/epoch=3-step=1610.ckpt",
    "resnet_vae": "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_35"
    "/checkpoints/last.ckpt",
}

rgb_ds = ds.rgb_cells
from old_code.models import VAE as ResNetVAE

the_model = "resnet_vae"
# the_model = 'resnet_vae_last_channel'
resnet_vae = ResNetVAE.load_from_checkpoint(models[the_model])
resnet_vae.cuda()
loader = DataLoader(rgb_ds, batch_size=1024, num_workers=8, pin_memory=True)
data = loader.__iter__().__next__()

##
start = time.time()
list_of_z = []
with torch.no_grad():
    for data in tqdm(loader, desc="embedding the whole validation set"):
        data = [d.to(resnet_vae.device) for d in data]
        expression, x, mask = data
        z = [zz.cpu() for zz in resnet_vae(x, mask)]
        list_of_z.append(z)
print(f"forwarning the data to the resnets: {time.time() - start}")
torch.cuda.empty_cache()

##

f = file_path("image_features.npy")
if True:
    # if False:
    l = []
    for zz in list_of_z:
        alpha, beta, mu, std, z = zz
        l.append(mu)
    mus = torch.cat(l, dim=0).numpy()
    np.save(f, mus)
mus = np.load(f)
##


a = ad.AnnData(mus)
sc.tl.pca(a)
sc.pl.pca(a)
##

random_indices = reproducible_random_choice(len(a), 10000)
##
b = a[random_indices]
##
print("computing umap... ", end="")
sc.pp.neighbors(b)
sc.tl.umap(b)
sc.tl.louvain(b)
print("done")
##
CH = 35

plt.figure()
l = b.obs["louvain"].tolist()
colors = list(map(int, l))
plt.scatter(
    b.obsm["X_umap"][:, 0],
    b.obsm["X_umap"][:, 1],
    s=1,
    c=b.X[:, CH]
    # c=colors,
    # cmap=matplotlib.cm.tab20,
)
# plt.xlim([10, 20])
# plt.ylim([0, 10])
plt.title(f"UMAP of image latent space colored by channel {CH} ({CHANNEL_NAMES[CH]})")
plt.show()
##

fig, ax = plt.subplots(figsize=(24, 14))
u = b.obsm["X_umap"]
for j, i in enumerate(tqdm(random_indices[:500], desc="scatterplot with images")):
    _, ome, mask, _ = ds[i]
    ome = ome.numpy()
    mask = torch.squeeze(mask, 0).numpy()
    im = OffsetImage(mask, zoom=0.7)
    ab = AnnotationBbox(im, u[j], xycoords="data", frameon=False)
    ax.add_artist(ab)
ax.set(xlim=(min(u[:, 0]), max(u[:, 0])), ylim=(min(u[:, 1]), max(u[:, 1])))
plt.tight_layout()
plt.show()
##
if True:
    # n_channels = ds[42][0].shape[0]
    # channels = list(range(n_channels))
    channels = [35]
    # channels = [10, 34, 35, 38]
    #     ax.autoscale()
    for c in tqdm(channels, desc="channels", position=0, leave=False):
        fig, ax = plt.subplots(figsize=(24, 14))
        ax.set(title=f"ch {c}")
        u = b.obsm["X_umap"]
        for j, i in enumerate(
            tqdm(
                random_indices[:2000],
                desc="scatterplot with images",
                position=1,
                leave=False,
            )
        ):
            _, ome, _, _ = ds[i]
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
loader = DataLoader(cells_ds, batch_size=1024, num_workers=8)
all_expressions = []
for data in tqdm(loader):
    expression, _, _ = data
    all_expressions.append(expression)
expressions = torch.cat(all_expressions, dim=0).numpy()
##
aa = ad.AnnData(expressions)
sc.tl.pca(aa)
sc.pl.pca(aa)
##
random_indices = reproducible_random_choice(len(expressions), 10000)
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
plt.title("showing expression clusters on umap of latent points")
plt.show()
##
CH = 35
plt.figure()
l = bb.obs["louvain"].tolist()
colors = list(map(int, l))
plt.scatter(
    bb.obsm["X_umap"][:, 0],
    bb.obsm["X_umap"][:, 1],
    s=1,
    # c=colors,
    # cmap=matplotlib.cm.tab20,
    c=bb.X[:, CH],
)
# plt.title("showing expression clusters on umap of expression points")
plt.title(f"UMAP of expression data colored by channel {CH} ({CHANNEL_NAMES[CH]})")
plt.show()
##
uu = bb.obsm["X_umap"]
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig, ax = plt.subplots(figsize=(24, 14))
u = b.obsm["X_umap"]
for j, i in enumerate(tqdm(random_indices[:500], desc="scatterplot with images")):
    _, ome, mask, _ = ds[i]
    ome = ome.numpy()
    mask = torch.squeeze(mask, 0).numpy()
    im = OffsetImage(mask, zoom=0.7)
    ab = AnnotationBbox(im, uu[j], xycoords="data", frameon=False)
    ax.add_artist(ab)
ax.set(xlim=(min(uu[:, 0]), max(uu[:, 0])), ylim=(min(uu[:, 1]), max(uu[:, 1])))
plt.tight_layout()
plt.show()
##
if True:
    # n_channels = ds[42][0].shape[0]
    # channels = list(range(n_channels))
    channels = [10, 34, 35, 38] + [20]
    #     ax.autoscale()
    for c in tqdm(channels, desc="channels", position=0, leave=False):
        fig, ax = plt.subplots(figsize=(24, 14))
        ax.set(title=f"ch {c}")
        # for the slides I commented this
        # u = b.obsm["X_umap"]
        u = bb.obsm["X_umap"]
        for j, i in enumerate(
            tqdm(
                random_indices[:2000],
                desc="scatterplot with images",
                position=1,
                leave=False,
            )
        ):
            _, ome, _, _ = ds[i]
            ome = ome[c, :, :].numpy()
            # mask = torch.squeeze(mask, 0).numpy()
            im = OffsetImage(ome, zoom=0.7)
            ab = AnnotationBbox(im, uu[j], xycoords="data", frameon=False)
            ax.add_artist(ab)
        ax.set(xlim=(min(uu[:, 0]), max(uu[:, 0])), ylim=(min(uu[:, 1]), max(uu[:, 1])))
        plt.tight_layout()
        plt.show()
##
from sklearn.neighbors import NearestNeighbors
import numpy as np

random_expressions = expressions[random_indices]
nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(random_expressions)
distances, indices = nbrs.kneighbors(random_expressions)
indices
##
# plot barplots of expression for nearest neighbors, of subsampled cells
plt.style.use("dark_background")
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
        ax.set(title=f"cell {i}, nn of {cell}")
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
            color=(0.3, 0.3, 0.3),
        )
        ax.scatter(bbb.obsm["X_umap"][nn, 0], bbb.obsm["X_umap"][nn, 1], s=1, color="w")
    plt.tight_layout()
    plt.show()
plt.style.use("default")
## study deviation of nearest neighbors from the selected cells
n_channels = ds[42][0].shape[0]
ab_for_cell = dict()
for cell in some_cells:
    list_of_squared_distances = []
    list_of_mse = []
    expression = random_expressions[cell]
    a = np.square(random_expressions - expression)
    b = np.mean(a, axis=1)
    ab_for_cell[cell] = (a, b)
##
for cell, (a, b) in ab_for_cell.items():
    plt.figure()
    plt.title(f"histogram of mse from cell {cell}")
    plt.hist(b)
    plt.show()

##
class Ecdf:
    def __init__(self, x):
        self.x = np.sort(x)
        self.y = np.arange(0, len(x), 1) / (len(x) - 1)

    def evaluate(self, t):
        i = np.searchsorted(self.x, t)
        if i == 0:
            return 0
        elif i == len(self.x):
            return 1
        else:
            #            print(f'{self.x[i - 1]} <= {t} <= {self.x[i]}')
            return self.y[i - 1]


##
a_ecdfs_for_cell = dict()
b_ecdf_for_cell = dict()

for cell in some_cells:
    a, b = ab_for_cell[cell]
    ecdfs = []
    for i in range(a.shape[1]):
        aa = a[:, i]
        ecdf = Ecdf(aa)
        ecdfs.append(ecdf)
    a_ecdfs_for_cell[cell] = ecdfs
    b_ecdf_for_cell[cell] = Ecdf(b)

##
# plot images for nearest neighbors, of one of the subsampled cells and for selected channels
def aaa(idx0, idx1=None):
    if idx1 is None:
        idx1 = idx0
    cell = some_cells[idx0]
    expression = random_expressions[idx1]
    selected_channels = [0, 37, 38, 3, 4, 5, 10, 35]
    fig, axes = plt.subplots(len(selected_channels), len(indices[42]), figsize=(30, 20))
    axes = axes.flatten()
    k = 0
    for channel in tqdm(selected_channels):
        for i in indices[cell]:
            ax = axes[k]
            _, ome, mask, _ = ds[random_indices[i]]
            im = ome[channel, :, :].numpy()
            ax.imshow(im)
            other_expression, _, _ = cells_ds[random_indices[i]]

            def to_simplex(x, left, right):
                return (x - left) / (right - left)

            aaaa = np.square(other_expression - expression)
            bbbb = np.mean(aaaa)
            aaaa = aaaa[channel]
            a_score = a_ecdfs_for_cell[cell][channel].evaluate(aaaa)
            b_score = b_ecdf_for_cell[cell].evaluate(bbbb)
            # a_for_cell, b_for_cell = ab_for_cell[idx1]
            # a_for_cell = a_for_cell[:, channel]
            # a_score = to_simplex(aaaa, np.min(a_for_cell), np.max(a_for_cell))
            # b_score = to_simplex(bbbb, np.min(b_for_cell), np.max(b_for_cell))

            ax.set(title=f"{a_score:0.3f}, {b_score:0.3f}")
            k += 1
            # plot contour of mask
            numpy_mask = mask.squeeze(0).numpy()
            contours = skimage.measure.find_contours(numpy_mask, 0.4)
            # ax.imshow(numpy_mask, alpha=0.4, cmap=matplotlib.cm.gray)
            for contour in contours:
                orange = list(map(lambda x: x / 255, (255, 165, 0)))
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=orange)
    fig.suptitle(f"idx0 = {idx0}, idx1 = {idx1}, channels = {selected_channels}")
    plt.tight_layout()
    plt.show()


aaa(0)
aaa(0, 1)
aaa(1)
aaa(2)

##
a = ad.AnnData(mus)
sc.tl.pca(a)
sc.pl.pca(a)

b = a[random_indices]

print("recomputing umap... ", end="")
sc.pp.neighbors(b)
sc.tl.umap(b)
sc.tl.louvain(b)
print("done")

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
plt.style.use("dark_background")


def bbb(idx):
    # n_channels = ds[42][0].shape[0]
    # channels = list(range(n_channels))
    channels = [35, 38]
    #     ax.autoscale()
    for c in tqdm(channels, desc="channels", position=0, leave=False):
        fig, ax = plt.subplots(figsize=(24, 14))
        ax.set(title=f"ch {c}")
        u = b.obsm["X_umap"]
        for j, i in enumerate(
            tqdm(
                random_indices[:2000],
                desc="scatterplot with images",
                position=1,
                leave=False,
            )
        ):
            _, ome, _, _ = ds[i]
            ome = ome[c, :, :].numpy()
            # mask = torch.squeeze(mask, 0).numpy()
            im = OffsetImage(ome, zoom=0.7, cmap="gray")
            ab = AnnotationBbox(im, u[j], xycoords="data", frameon=False)
            ax.add_artist(ab)
        ax.set(xlim=(min(u[:, 0]), max(u[:, 0])), ylim=(min(u[:, 1]), max(u[:, 1])))

        cell = some_cells[idx]
        for cell_index in indices[cell]:
            _, ome, _, _ = ds[random_indices[cell_index]]
            ome = ome[c, :, :].numpy()
            # mask = torch.squeeze(mask, 0).numpy()
            im = OffsetImage(ome, zoom=0.7, cmap="viridis")
            ab = AnnotationBbox(im, u[cell_index], xycoords="data", frameon=False)
            ax.add_artist(ab)

        plt.tight_layout()
        plt.show()


bbb(0)
plt.style.use("default")
