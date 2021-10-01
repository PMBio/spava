##

import importlib
import math

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
from torch.utils.data import DataLoader
from tqdm import tqdm

import analyses.essentials
from data2 import CellDataset
from data2 import RGBCells

importlib.reload(analyses.essentials)
from analyses.essentials import save_plot, DARK_GREEN
from utils import reproducible_random_choice
import skimage.measure
from data2 import CHANNEL_NAMES
from analyses.aa_reconstruction_benchmark.aa_ad_reconstruction import Space, transform

SPLIT = "validation"
rgb_ds = RGBCells(SPLIT)
cells_ds = CellDataset(SPLIT)
BATCH_SIZE = 1024

##
loader = DataLoader(
    rgb_ds,
    batch_size=BATCH_SIZE,
    num_workers=16,
)
l0 = []
l1 = []
l2 = []
for i, data in tqdm(enumerate(loader), desc="merging cell ds", total=len(loader)):
    expression, _, _ = data
    l0.append(expression)
    l1.append(np.arange(len(expression)) + i * BATCH_SIZE)
expressions = np.concatenate(l0, axis=0)
are_perturbed = np.concatenate(l1, axis=0)

##
# d = {0: 'subcellular', 37: 'subcellular', 38: 'subcellular',
#      3: 'boundary', 4: 'boundary', 5: 'boundary',
#      10: 'both', 35: 'both'}

CH = 20
x = expressions[:, CH]
y = np.concatenate(l1)

# import pickle
# pickle.dump(x, open(file_path('to_del.pickle'), 'wb'))
# x = pickle.load(open(file_path('to_del.pickle'), 'rb'))

# it is not a problem for our conclusion, but be aware that when transforming back from asinh_mean to raw_sum for
# something like the expression give by a RGBCell dataset, then in the case in which the original cell was larger
# than the tile, the expression will be different from the one that would have been computed from the raw data
x = x.reshape(-1, 1)
x = transform(x, from_space=Space.scaled_mean, to_space=Space.raw_sum, split=SPLIT)
x = np.squeeze(x, 1)

qs = [0.01, 0.5, 0.99]
plt.figure()
plt.hist(x, bins=100, color=DARK_GREEN)
for q in qs:
    xq = np.quantile(x, q)
    plt.axvline(x=xq, c="r")
plt.title(f"channel {CH} ({CHANNEL_NAMES[CH]}), quantiles: {qs}")
plt.xlabel("expression")
plt.ylabel("count")
ylim = plt.gca().get_ylim()
plt.ylim(ylim[0], ylim[1] * 1.1)
plt.show()

##
sorted_xyz = np.array(sorted(zip(x.tolist(), y.tolist()), key=lambda x: x[0]))
sorted_x = sorted_xyz[:, 0]
sorted_y = sorted_xyz[:, 1].astype(int)

for q in tqdm(qs, desc="plotting similar images"):
    xq = np.quantile(sorted_x, q)
    n = np.searchsorted(sorted_x, xq)
    s = 81
    # we don't believe in numerical issues
    k = int(round(math.sqrt(s)))
    ss = slice(max(0, n - s // 2), min(n + s // 2 + 1, len(sorted_x)))
    similar = list(zip(sorted_x[ss], sorted_y[ss]))
    vmin = 100000
    vmax = 0
    for x, y in similar:
        expression, ome, mask = rgb_ds[y]
        vmin, vmax = min(ome[CH].min().item(), vmin), max(ome[CH].max().item(), vmax)
    axes = plt.subplots(k, k, figsize=(15, 15))[1].flatten()
    plt.suptitle(f"quantile {q}: {similar[0][0]} - {similar[-1][0]}")
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        if i >= len(similar):
            continue
        x, y = similar[i]
        expression, ome, mask = rgb_ds[y]
        # this works only if not transforming Space.raw_sum
        # assert np.isclose(x, expression[c])
        ax.imshow(ome[CH, :, :], vmin=vmin, vmax=vmax)
        numpy_mask = np.squeeze(mask.numpy(), 0)
        contours = skimage.measure.find_contours(numpy_mask, 0.4)
        for contour in contours:
            orange = list(map(lambda x: x / 255, (255, 165, 0)))
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color="r")
    plt.tight_layout()
    plt.show()
##
"""
VISIUM
"""
import scanpy as sc
import squidpy as sq

import numpy as np

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# load the pre-processed dataset
img = sq.datasets.visium_hne_image()
adata = sq.datasets.visium_hne_adata()

##
plt.style.use("dark_background")
for s in ["Olfm1", "Plp1", "Itpka"]:
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    sc.pl.spatial(adata, color=s, alpha_img=0.0, show=False, ax=ax)
    plt.show()
plt.style.use("default")

##
"""
PREPROCESSING
"""
sums = transform(
    expressions, from_space=Space.scaled_mean, to_space=Space.raw_sum, split=SPLIT
)
means = transform(
    expressions, from_space=Space.scaled_mean, to_space=Space.raw_mean, split=SPLIT
)
transformed = transform(
    expressions, from_space=Space.scaled_mean, to_space=Space.asinh_mean, split=SPLIT
)

##
ii = reproducible_random_choice(len(expressions), 200)


def plot_merged(merged, title):
    plt.figure(figsize=(5, 10))
    plt.imshow(merged[ii, :])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("39 proteins")
    plt.ylabel("900k cells")
    plt.title(title)
    save_plot(f"preprocessing_{title}.png")
    plt.show()


plot_merged(sums, "expression (sum)")
plot_merged(means, "expression (mean)")
plot_merged(transformed, "expression (mean + arcsinh)")
plot_merged(expressions, "expression (mean + arcsinh + scaled)")

##
"""
VARIANCE STABILIZATION
"""


def mean_std_plot(raw_counts):
    means = np.mean(raw_counts, axis=0)
    stds = np.std(raw_counts, axis=0)
    light_green = np.array([70, 155, 82]) / 255
    dark_green = np.array([53, 121, 86]) / 255
    plt.style.use("default")
    plt.figure()
    plt.scatter(means, stds, s=1, label="cells", color=dark_green)
    plt.axis("equal")
    # plt.xlim(1e-4, 1e3)
    # plt.ylim(1e-4, 1e3)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("gene-wise mean")
    plt.ylabel("gene-wise std")
    plt.title("mean-std relationship")
    x = np.linspace(0, 4)
    # x = np.logspace(-4, 8)
    # plt.plot([1e-4, 1e3], [1e-4, 1e3], c='r')
    plt.plot(x, x, c="k", linewidth=1, label="y=x")
    plt.legend(scatterpoints=3, markerscale=1.5)
    # plt.plot(x, x + 100 * x ** 2, c='r', linewidth=1)
    print(means, stds)


mean_std_plot(transformed)
save_plot("imc_scaled_mean_std.png")
plt.show()

##
