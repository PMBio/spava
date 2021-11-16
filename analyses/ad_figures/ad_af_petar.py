##


import math

import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import skimage.measure

from analyses.essentials import save_plot
from sklearn.decomposition import PCA
from data2 import CenterFilteredDataset
from data2 import OmeDataset, quantiles_for_normalization
from data2 import RGBCells

##
# x = ds[0]
#
# plt.figure(figsize=(7, 7))
# x = np.arcsinh(x.numpy())
# x /= quantiles_for_normalization
# reducer = PCA(3)
# old_shape = x.shape
# x = x.reshape(-1, old_shape[-1])
# pca = reducer.fit_transform(x)
# a = np.min(pca, axis=0)
# b = np.max(pca, axis=0)
# pca = (pca - a) / (b - a)
# pca.shape = (old_shape[0], old_shape[1], 3)
# plt.imshow(pca, cmap="Greys_r")
# plt.axis("off")
# save_plot(f"imc_3_channels{i}.png")
# plt.show()
#
# ##

plt.style.use("dark_background")

ds = OmeDataset(split="train")
# for j in range(10, 25):
j = 22
x = ds[j]
x = np.arcsinh(x.numpy())
x /= quantiles_for_normalization

from data2 import CHANNEL_NAMES

d = {
    "tumor": ["CK5", "CK7", "CK14"],
    "immune": ["CD3", "CD20", "CD45", "CD68"],
    "hist": ["H3tot"],
    # "nuclei": ["DNA1"],
    # 'necrosis': ['cPARP_cCasp3'],
}

orange = np.array([242, 146, 39]) / 255
purple = np.array([226, 87, 242]) / 255
blue = np.array([39, 185, 242]) / 255
green = np.array([144, 242, 51]) / 255
colors = {
    "tumor": purple,
    "immune": orange,
    "hist": blue.tolist() + [0.5],
    "nuclei": green,
    # 'necrosis': 'yellow'
}

plt.figure(figsize=(7, 7))
ax = plt.gca()
for k, v in d.items():
    c = colors[k]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", [(0.0, 0.0, 0.0, 0.0), c]
    )
    for vv in v:
        i = CHANNEL_NAMES.index(vv)
        # xx = x[:200, :200, i]
        xx = x[:, :, i]
        norm = plt.Normalize(0, np.max(xx))
        ax.imshow(np.moveaxis(xx, 0, 1), cmap=cmap)
        # ax.set_axis_off()
        ax.set(xlabel='', xticks=[], ylabel='', yticks=[])
plt.title('Spaital protein profiling')

from data2 import ExpressionFilteredDataset
ds2 = ExpressionFilteredDataset('train')
ds0 = CenterFilteredDataset("train")
xy = ds0[j]
if True:
    expressions = ds2[j]
    from data2 import IndexInfo
    ii = IndexInfo('train')
    begin = ii.filtered_begins[j]
    end = ii.filtered_ends[j]
    ds1 = RGBCells("train")
    from tqdm import tqdm
    for jj in tqdm(range(begin, end)):
        expression, ome, mask = ds1[jj]
        numpy_mask = np.squeeze(mask.numpy(), 0)
        contours = skimage.measure.find_contours(numpy_mask, 0.4)
        for contour in contours:
            orange = list(map(lambda x: x / 255, (255, 165, 0)))
            x, y = xy[jj - begin]
            e = expression
            e /= np.max(expressions, axis=0)
            scores = {k: 0 for k in d.keys()}
            for k, v in d.items():
                for vv in v:
                    idx = CHANNEL_NAMES.index(vv)
                    scores[k] = max(scores[k], e[idx])
            highest = max(scores, key=lambda k: scores.get(k))
            color = colors[highest]
            ax.plot(contour[:, 0] + x - 16, contour[:, 1] + y - 16, linewidth=1, color=color)
    plt.title('Visualizing cells')

if True:
    from matplotlib_scalebar.scalebar import ScaleBar

    scalebar = ScaleBar(1, "um", length_fraction=0.25, box_color='black', color='white', location='lower right',
                        fixed_value=100)
    ax.add_artist(scalebar)
    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], marker='o', color=colors['tumor'], markersize=10, lw=0),
                    Line2D([0], [0], marker='o', color=colors['immune'], markersize=10, lw=0),
                    Line2D([0], [0], marker='o', color=colors['hist'], markersize=10, lw=0)]

    ax.legend(custom_lines, ['Tumor', 'Immune', 'Stroma'])

if True:
    cmap = matplotlib.cm.get_cmap('Blues')
    from graphs import get_graph_file

    f = get_graph_file('knn', 'train', j)
    import torch

    data = torch.load(f)

    from torch_geometric.utils.convert import to_networkx

    g = to_networkx(data, edge_attrs=["edge_attr"], node_attrs=["regions_centers"])
    positions = {i: d["regions_centers"] for i, d in g.nodes(data=True)}
    weights = np.array([x[2]["edge_attr"] for x in g.edges(data=True)])
    edges_to_plot = [(e[0], e[1]) for e in g.edges(data=True) if e[2]["edge_attr"] > 0]

    import networkx
    node_size = 0
    node_colors = 'black'
    edge_cmap = cmap
    networkx.drawing.nx_pylab.draw_networkx(
        g,
        pos=positions,
        edgelist=edges_to_plot,
        node_size=node_size,
        # linewidths=linewidths,
        with_labels=False,
        width=0.5,
        arrows=False,
        node_color=node_colors,
        edge_color=weights,
        edge_cmap=cmap,
        edge_vmin=np.min(weights),
        edge_vmax=np.max(weights),
        ax=ax,
    )
    plt.title('Physical proximity graph')

x = ds[j]
ax.set(xlim=[0, x.shape[0]], ylim=[0, x.shape[1]])

# for i in range(len(xy)):
#     xx_i = xy[i, 1]
#     yy_i = xy[i, 0]
#     for j in range(len(xy)):
#         xx_j = xy[j, 1]
#         yy_j = xy[j, 0]
#         # if (
#         #     150 < xx_i < 200
#         #     and 150 < yy_i < 200
#         #     and 150 < xx_j < 200
#         #     and 150 < yy_j < 200
#         # ):
#         d = math.sqrt((xx_i - xx_j) ** 2 + (yy_i - yy_j) ** 2)
#         if d < 40:
#             plt.plot(
#                 [xx_i, xx_j],
#                 [yy_i, yy_j],
#                 zorder=1,
#                 color=cmap(np.random.rand(1).item()),
#                 alpha=0.5
#             )
plt.tight_layout()
plt.show()

##

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
