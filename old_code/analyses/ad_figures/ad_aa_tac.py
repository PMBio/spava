##
import scipy.stats

from splits import train, validation, test
from old_code.data2 import RGBCells
import skimage.measure

import math
import os
import matplotlib.cm
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from old_code.data2 import OmeDataset, quantiles_for_normalization, file_path
from old_code.data2 import CenterFilteredDataset
from analyses.essentials import save_plot


##
ds = OmeDataset(split="train")
x = ds[0]
for i in range(3):
    plt.figure(figsize=(7, 7))
    plt.imshow(
        np.arcsinh(x[:200, :200, i]) / quantiles_for_normalization[i], cmap="Greys_r"
    )
    plt.axis("off")
    save_plot(f"imc_3_channels{i}.png")
    plt.show()

##

ds0 = CenterFilteredDataset("train")
for i in range(3):
    x = ds[0][:, :, i]
    xy = ds0[0]

    plt.figure(figsize=(7, 7))
    plt.imshow(np.arcsinh(x), cmap="Greys_r")
    plt.scatter(xy[:, 1], xy[:, 0], c="r", s=10)
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.axis("off")
    save_plot(f"centers_{i}.png")
    plt.show()

##
plt.figure(figsize=(7, 7))
# plt.imshow(np.arcsinh(x), cmap="Greys_r")
k = (6 / 255, 158 / 255, 73 / 255)
k = "k"
c = [k] * len(xy)
c[1094] = "r"
s = list(map(str, range(len(xy))))

cmap = matplotlib.cm.viridis
for i in range(len(xy)):
    xx_i = xy[i, 1]
    yy_i = xy[i, 0]
    for j in range(len(xy)):
        xx_j = xy[j, 1]
        yy_j = xy[j, 0]
        if (
            150 < xx_i < 200
            and 150 < yy_i < 200
            and 150 < xx_j < 200
            and 150 < yy_j < 200
        ):

            d = math.sqrt((xx_i - xx_j) ** 2 + (yy_i - yy_j) ** 2)
            if d < 20:
                plt.plot(
                    [xx_i, xx_j],
                    [yy_i, yy_j],
                    zorder=1,
                    color=cmap(np.random.rand(1).item()),
                )
    # print(f'x = {x}, y = {y}, s[i] = {s[i]}')
    # plt.text(x, y, s[i])
plt.scatter(xy[:, 1], xy[:, 0], c=c, s=80, zorder=2)
plt.xlim([150, 200])
plt.ylim([150, 200])
plt.axis("off")
save_plot("centers_nn.png")
plt.show()

##

ds = RGBCells("train")
_, ome, mask = ds[1094]
numpy_mask = np.squeeze(mask.numpy(), 0)
# axes = plt.subplots(1, 3, figsize=(8, 5))[1].flatten()
axes = plt.subplots(5, 8, figsize=(8, 5))[1].flatten()
# ax = axes[0]scipy
# ax.imshow(np.squeeze(mask, 0))
# ax.set_axis_off()
n = 16
# ax.scatter(n, n, c='r', s=6)
contours = skimage.measure.find_contours(numpy_mask, 0.4)
# for i in range(0, 3):
for i in range(0, 39):
    ax = axes[i]
    ax.imshow(ome[i, :, :], cmap="Greys_r")
    ax.invert_yaxis()
    ax.scatter(n, n, c="r", s=2)
    ax.set_axis_off()

    for contour in contours:
        orange = list(map(lambda x: x / 255, (255, 165, 0)))
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="r")
    # ax.set_title(f'gene {i}')
axes[39].set_axis_off()
plt.tight_layout()
save_plot("images.png")
plt.show()

##
total = len(train) + len(validation) + len(test)
print("patients:", total)
print("train:", len(train) / total)
print("validation:", len(validation) / total)
print("test:", len(test) / total)
print(
    "cells:",
    len(RGBCells("train")) + len(RGBCells("validation")) + len(RGBCells("test")),
)

##
ds = RGBCells("train")
_, ome, mask = ds[1094]
numpy_mask = np.squeeze(mask.numpy(), 0)
# axes = plt.subplots(1, 3, figsize=(8, 5))[1].flatten()
axes = plt.subplots(5, 8, figsize=(8, 5))[1].flatten()
# ax = axes[0]scipy
# ax.imshow(np.squeeze(mask, 0))
# ax.set_axis_off()
n = 16
# ax.scatter(n, n, c='r', s=6)
contours = skimage.measure.find_contours(numpy_mask, 0.4)
# for i in range(0, 3):
for i in range(0, 39):
    ax = axes[i]
    t = np.tile(np.random.rand(1), (32, 32, 3))
    ax.imshow(t)
    ax.invert_yaxis()
    ax.text(n, n - 1, f"{t[0, 0, 0].item():0.2f}", ha="center", va="center")
    # ax.scatter(n, n, c='r', s=2)
    # ax.set_axis_off()
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth("1")
    ax.set_xticks([])
    ax.set_yticks([])

    # for contour in contours:
    #     orange = list(map(lambda x: x / 255, (255, 165, 0)))
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
    # ax.set_title(f'gne {i}')
axes[39].set_axis_off()
plt.tight_layout()
save_plot("images.png")
plt.show()

##
from old_code.data2 import PerturbedCellDataset
from analyses.essentials import merge_perturbed_cell_dataset

x = merge_perturbed_cell_dataset(PerturbedCellDataset("validation"))

##
ii = np.random.choice(len(x[0]), 200, replace=False)
imc = x[0][ii]

plt.figure(figsize=(5, 10))
plt.imshow(imc)
plt.xticks([])
plt.yticks([])
plt.xlabel("39 proteins")
plt.ylabel("900k cells")
plt.title("IMC expression")
save_plot("imc_expression.png")
plt.show()

##
"""
SCANPY
"""

import numpy as np
import scanpy as sc

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

results_file = "/data/l989o/data/scrna_seq/pbmc3k/write/pbmc3k.h5ad"

adata = sc.read_10x_mtx(
    "/data/l989o/data/scrna_seq/pbmc3k/filtered_gene_bc_matrices/hg19/",  # the directory with the `.mtx` file
    var_names="gene_symbols",  # use gene symbols for the variable names (variables-axis index)
    cache=True,
)  # write a cache file for faster subsequent reading

adata.var_names_make_unique()

adata.raw = adata.copy()

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var["mt"] = adata.var_names.str.startswith(
    "MT-"
)  # annotate the group of mitochondrial genes as 'mt'
# sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# adata = adata[adata.obs.n_genes_by_counts < 2500, :]
# adata = adata[adata.obs.pct_counts_mt < 5, :]

sc.pp.normalize_total(adata, target_sum=1e4)

sc.pp.log1p(adata)

x = adata.X.todense()
##
# sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# adata = adata[:, adata.var.highly_variable]
# sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
# sc.tl.paga(adata)
sc.tl.leiden(adata)
sc.tl.umap(adata)

##
sc.pl.umap(adata, color=[0], show=False, s=6)
ax = plt.gca()
ax.set_title("latent space")
save_plot("2d_latent_space.png")
plt.show()


##
cols = 400
rows = round(cols / x.shape[1] * x.shape[0])
print(rows)
ii0 = np.random.choice(x.shape[0], rows, replace=False)
ii1 = np.random.choice(x.shape[1], cols, replace=False)
max(ii0)
max(ii1)
x.shape
sc_data = x[ii0, :][:, ii1]
sc_data.shape

##
plt.figure(figsize=(10, 5))
plt.imshow(sc_data)
plt.xticks([])
plt.yticks([])
plt.xlabel("14k genes")
plt.ylabel("3k cells")
plt.title("scRNA-seq")
save_plot("sc_expression.png")
plt.show()

##
raw_sc = adata.raw.X.todense()
import scipy.stats

##
ind = np.argpartition(np.max(raw_sc, axis=0).A1, -4)[-4:]
print(ind)

##
for gene in ind:
    # measurements = np.random.normal(loc=20, scale=5, size=100)
    measurements = raw_sc[:, gene].A1
    scipy.stats.probplot(measurements, dist=scipy.stats.nbinom(10, 0.5), plot=plt)
    plt.show()

##
raw_sc = raw_sc.getA()
raw_sc_backup = raw_sc.copy()
raw_sc = raw_sc_backup.copy()
##
s = np.sum(raw_sc, axis=1)
b = s > 0
raw_sc = (raw_sc[b, :].T / s[b] * 10000).T
print(s.shape)
print(raw_sc.shape)
# raw_sc += 1

##
def mean_std_plot(raw_counts):
    means = np.mean(raw_counts, axis=0)
    stds = np.std(raw_counts, axis=0)
    light_green = np.array([70, 155, 82]) / 255
    dark_green = np.array([53, 121, 86]) / 255
    plt.style.use("default")
    plt.figure()
    plt.scatter(means, stds, s=1, label="cells", color=dark_green)
    plt.axis("equal")
    plt.xlim(1e-4, 1e3)
    plt.ylim(1e-4, 1e3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("gene-wise mean")
    plt.ylabel("gene-wise std")
    plt.title("mean-std relationship")
    x = np.logspace(-4, 8)
    # plt.plot([1e-4, 1e3], [1e-4, 1e3], c='r')
    plt.plot(x, x, c="k", linewidth=1, label="y=x")
    plt.legend(scatterpoints=3, markerscale=1.5)
    # plt.plot(x, x + 100 * x ** 2, c='r', linewidth=1)


##
mean_std_plot(raw_sc)
save_plot("scrna_seq_mean_std.png")
plt.show()

##
"""
EXPRESSION RAW DATA
"""
from old_code.data2 import SumFilteredDataset
from tqdm import tqdm

ds = SumFilteredDataset("validation")
l0 = []
for i in tqdm(range(len(ds))):
    expression = ds[i]
    l0.append(expression)
expressions = np.concatenate(l0, axis=0)

##
mean_std_plot(expressions)
l = [1e1, 1e4]
plt.xlim(l)
plt.ylim(l)
save_plot("imc_mean_std.png")
plt.show()

##
"""
DENOISING
"""

pop_a = 100 + np.random.normal(0, 20, size=(10,))
pop_a
pop_b = 120 + np.random.normal(0, 50, size=(7,))
highest = np.max((pop_a.max(), pop_b.max())) + 10
##

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
l = np.arange(len(pop_a))
plt.bar(l, pop_a, color=dark_green, width=0.8)
plt.xticks(l, list(map(lambda x: f"cell {x}", l)), rotation=60)
plt.ylabel("NCAM1 expression")
plt.ylim(0, highest)
plt.gca().set_aspect(0.05)
plt.title("population A")

plt.subplot(1, 2, 2)
l = np.arange(len(pop_b))
plt.bar(l, pop_b, color=dark_green, width=0.8)
plt.xticks(l, list(map(lambda x: f"cell {x + len(pop_a)}", l)), rotation=60)
plt.ylabel("NCAM1 expression")
plt.ylim(0, highest)
plt.gca().set_aspect(0.05)
plt.title("population B")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
save_plot("populations.png")
plt.show()

##
from umap import UMAP

reducer = UMAP(n_components=3, verbose=True)
u = reducer.fit_transform(raw_sc)
u.shape

##
from mpl_toolkits.mplot3d import Axes3D

x = u.copy()
mins = np.min(x, axis=0)
maxs = np.max(x, axis=0)
x = (x - mins) / (maxs - mins)

list_of_colors = [x[i, :] for i in range(len(x))]

fig = plt.figure(figsize=(9, 9))
ax = Axes3D(fig)

ax.set_xlabel("UMAP 0")
ax.set_ylabel("UMAP 1")
ax.set_zlabel("UMAP 2")
ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=raw_sc[:, ind[np.argmax(ind)]], s=6)
# ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=list_of_colors, s=6)

os.makedirs(file_path("frames_3d"), exist_ok=True)
for ii in tqdm(range(0, 360, 2), desc="generating frames"):
    ax.view_init(elev=30.0, azim=ii)
    plt.savefig(file_path(f"frames_3d/movie{ii:03d}.png"))
plt.show()

##
f = file_path("frames_3d/movie*.png")
f_out = file_path("frames_3d.mp4")
cmd = f'ffmpeg -y -framerate 30 -pattern_type glob -i "{f}" -pix_fmt yuv420p {f_out}'
subprocess.check_output(cmd, shell=True)

##
"""
IMPUTATION BENCHMARK
"""
x = adata.X[:40, :70].todense()

##
plt.figure(figsize=(12, 12), dpi=300)

plt.subplot(3, 3, 1)
plt.imshow(x)
plt.xlabel("genes")
plt.ylabel("cells")
plt.xticks([])
plt.yticks([])
plt.title("expression")
# vmin, vmax = plt.gci().get_clim()

plt.subplot(3, 3, 2)
plt.xlabel("genes")
plt.ylabel("cells")
plt.xticks([])
plt.yticks([])
plt.title(r"$\rho$ values")
plt.imshow(x / 1.5)
plt.clim(np.min(x), np.max(x))

plt.subplot(3, 3, 3)
plt.xlabel("genes")
plt.xticks([])
plt.yticks([])
plt.title(r"$\theta$ values (dispersion)")
plt.imshow(np.random.rand(1, x.shape[1]))

plt.subplot(3, 3, 4)
plt.xlabel("genes")
plt.ylabel("cells")
plt.xticks([])
plt.yticks([])
plt.title(r"$\delta$ values (dropouts)")
plt.imshow(np.exp(10 * np.random.rand(*x.shape)))

plt.subplot(3, 3, 5)
xx = x.copy()
xx = xx + (-0.5 + np.random.rand(*xx.shape)) * (np.max(xx) - np.min(xx)) / 3.5
xx = np.clip(xx, np.min(x), np.max(x))
plt.imshow(xx)
plt.xlabel("genes")
plt.ylabel("cells")
plt.xticks([])
plt.yticks([])
plt.title("predictions")
# plt.gci().set_clim(vmin, vmax)

perturbed_entries = np.random.rand(*x.shape) < 0.1
alpha_red = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])

plt.subplot(3, 3, 6)
plt.imshow(x)
plt.xlabel("genes")
plt.ylabel("cells")
plt.xticks([])
plt.yticks([])
plt.title("expression")
plt.imshow(alpha_red[perturbed_entries.astype(int)])

plt.subplot(3, 3, 7)
plt.imshow(xx)
plt.xlabel("genes")
plt.ylabel("cells")
plt.xticks([])
plt.yticks([])
plt.title("predictions")
plt.imshow(alpha_red[perturbed_entries.astype(int)])

plt.tight_layout()

save_plot("imputation_showcase.png")
plt.show()
