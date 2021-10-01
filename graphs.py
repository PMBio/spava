import math
import os
import random
import time
import colorsys
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx
import numpy as np
import numpy.linalg
import torch
import torch_geometric
from scipy.ndimage.morphology import binary_dilation
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm

from data2 import (
    FilteredMasksRelabeled,
    file_path,
    CenterFilteredDataset,
    PerturbedCellDataset,
)
from analyses.essentials import merge_perturbed_cell_dataset

GRAPH_CONTACT_PIXELS = 4
GRAPH_GAUSSIAN_R_THRESHOLD = 0.1
GRAPH_GAUSSIAN_L = 400
GRAPH_KNN_K = 10
GRAPH_KNN_MAX_DISTANCE = 300
GRAPH_KNN_SUBGRAPH_K = 5
SUBGRAPH_RADIUS = 50

COMPUTE_GRAPH_FILES = False
COMPUTE_SUBGRAPH_DATASET = True
COMPUTE_OPTIMIZED_SUBGRAPH_DATASET = False
COMPUTE_PERTURBED_DATASET = False
COMPUTE_PERTURBED_CELLS_DATASET = False
PLOT = False
TEST = False

m = __name__ == "__main__"


def plot_imc_graph(
    data,
    split: str = None,
    ome_index=None,
    custom_ax=None,
    plot_expression: bool = False,
):
    ##
    # ##
    # data = load_graph('gaussian', 'validation', 0)
    # custom_ax = None
    # # ----------

    g = to_networkx(data, edge_attrs=["edge_attr"], node_attrs=["regions_centers"])
    positions = {i: d["regions_centers"] for i, d in g.nodes(data=True)}
    weights = np.array([x[2]["edge_attr"] for x in g.edges(data=True)])
    edges_to_plot = [(e[0], e[1]) for e in g.edges(data=True) if e[2]["edge_attr"] > 0]
    # f = networkx.Graph()
    # f_edges = filter(lambda x: x[2]['edge_attr'] > 0, g.edges(data=True))
    # f.add_edges_from(f_edges)
    if custom_ax is None:
        fig = plt.figure(figsize=(16, 9))
        ax = plt.gca()
    else:
        ax = custom_ax
    ax.set_facecolor((0.0, 0.0, 0.0))
    # ax.set_facecolor((1.0, 0.47, 0.42))
    greys = matplotlib.cm.get_cmap("Greys_r")
    node_colors = "black"

    if ome_index is not None:
        assert split is not None
        masks_ds = FilteredMasksRelabeled(split)
        if plot_expression:
            colors = [
                [r, r, r] for _ in range(10000) for r in [20 * random.random() / 100]
            ]
            colors[0] = (0.3, 0.3, 0.3)
        else:
            colors = [
                colorsys.hsv_to_rgb(
                    5 / 360, 58 / 100, (60 + random.random() * 40) / 100
                )
                for _ in range(10000)
            ]
            colors[0] = colorsys.hsv_to_rgb(5 / 360, 58 / 100, 63 / 100)

        masks = masks_ds[ome_index]
        if plot_expression:
            x = data.x
            from sklearn.decomposition import PCA

            reducer = PCA(3)
            pca = reducer.fit_transform(x)
            a = np.min(pca, axis=0)
            b = np.max(pca, axis=0)
            pca = (pca - a) / (b - a)
            is_near = data.is_near
            cells_to_color = np.arange(masks.max() + 1)[1:][is_near]

        colors = np.array(colors)
        if plot_expression:
            colors[cells_to_color] = pca
            node_colors = pca
        ax.imshow(colors[np.moveaxis(masks, 0, 1)], aspect="auto")
        # ax = plt.gca()
    # else:
    #     ax = None
    # node_colors = ["#ffffff"] * len(positions)
    if hasattr(data, "center_index") and plot_expression:
        node_colors[data.center_index] = [1.0, 1.0, 1.0]
        node_size = [0] * len(cells_to_color)
        node_size[data.center_index] = 100
    else:
        node_size = 10
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
        edge_cmap=greys,
        edge_vmin=np.min(weights),
        edge_vmax=np.max(weights),
        ax=ax,
    )
    if ome_index is not None and plot_expression:
        masks_copy = masks.copy()
        take = np.array([False] * 10000)
        take[cells_to_color] = True
        content = take[masks_copy]
        p0 = np.sum(content, axis=0)
        p1 = np.sum(content, axis=1)
        (w0,) = np.where(p0 > 0)
        (w1,) = np.where(p1 > 0)
        a0 = w0[0]
        b0 = w0[-1] + 1
        a1 = w1[0]
        b1 = w1[-1] + 1
        plt.xlim((a1, b1))
        plt.ylim((a0, b0))
    if custom_ax is None:
        plt.show()
    # return ax
    # # -------------
    # plot_hist(data)
    # ##


##
def plot_hist(data):
    x = data.edge_attr.numpy()
    plt.figure()
    plt.hist(x)
    plt.show()


def get_graph_file(graph_method: str, split: str, ome_index: int):
    f = file_path("graphs")
    os.makedirs(f, exist_ok=True)
    f = os.path.join(f, graph_method)
    os.makedirs(f, exist_ok=True)
    f = os.path.join(f, f"{split}_{ome_index}.torch")
    return f


def validate_graph_method(graph_method: str):
    assert graph_method in ["gaussian", "knn", "contact", None]


def compute_graphs(graph_method: str, split: str, ome_index: int):
    validate_graph_method(graph_method)
    ds_centers = CenterFilteredDataset(split=split)
    ds_masks = FilteredMasksRelabeled(split=split)
    # if ome_index > 0:
    #     return
    regions_centers = ds_centers[ome_index]
    edges = []
    weights = []
    extra_data = []
    if graph_method == "gaussian":
        tree = cKDTree(regions_centers)
        for i in range(len(regions_centers)):
            a = np.array(regions_centers[i])
            r_threshold = GRAPH_GAUSSIAN_R_THRESHOLD
            # in order to be in the graph, two cells must have their center closer than d_threshold microns (i.e.
            # d_threshold pixels in the image)
            l = GRAPH_GAUSSIAN_L
            d_threshold = math.sqrt(0 - math.log(r_threshold) * l)
            neighbors = tree.query_ball_point(a, d_threshold, p=2)
            for j in neighbors:
                if i > j:
                    continue
                b = np.array(regions_centers[j])
                c = a - b
                r = np.exp(-np.dot(c, c) / l)
                assert r >= r_threshold, (r, r_threshold)
                edges.append([i, j])
                edges.append([j, i])
                weights.append(r)
                weights.append(r)
    elif graph_method == "knn":
        k = GRAPH_KNN_K
        kk = min(k, len(regions_centers))
        neighbors = NearestNeighbors(n_neighbors=kk, algorithm="ball_tree").fit(
            regions_centers
        )
        distances, indices = neighbors.kneighbors(regions_centers)
        assert np.all(np.diff(distances, axis=1) >= 0)
        # check that every element is a knn of itself, and this is the first neighbor
        assert all(indices[:, 0] == np.array(range(len(regions_centers))))
        edges = []
        weights = []
        # TODO FIX
        extra_data.append(indices[:, :GRAPH_KNN_K])
        for i in range(indices.shape[0]):
            for j in range(1, indices.shape[1]):
                edge = [i, indices[i, j]]
                d = distances[i, j]
                # weight = 1
                # weight = 1 / (distances[i, j] ** 2 + 1)
                weight = d
                if d < GRAPH_KNN_MAX_DISTANCE:
                    edges.append(edge)
                    weights.append(weight)
    elif graph_method == "contact":
        # overview of the algorithm for building a graph which connects cells that are closer than a certain
        # number of pixels
        # 1. we take the masks of each image and for each mask (1 mask = 1 cell) we dilate the
        # mask by a number of pixels
        # 2. we identify those pixels that, after dilation, belong to the background
        # and to more than one mask the total number of this pixels is small compared to the original number of
        # pixels in the image
        # 3. we now create a list for each of those pixels, and check for each mask,
        # if the dilated mask contains that pixel if so, we add the mask id to the corresponding list
        # 4. now, we use those lists to find those cells that have at least one pixel (in the dilated mask) which
        # overlap with another cell
        #
        # a note on performance: while this approach is way faster than a bruteforce one, it is still slow (6 hours and 30 minutes)
        # to improve the performance, in steps 1-2 I am computing a bounding box and considering only the pixels in the bounding box;
        # also, in steps 3 I have tried using interval trees and the bounding boxes computed before to iterate only on those masks that contain the point being considered at each step
        # but this didn't improve the performance
        masks = ds_masks[ome_index]
        mask_ids = set(np.unique(masks))
        mask_ids.remove(0)

        def create_binary_circle(r: int):
            l = 2 * r + 1
            c = r
            x = np.zeros((l, l), dtype=bool)
            for i in range(l):
                for j in range(l):
                    if (i - c) ** 2 + (j - c) ** 2 <= r ** 2:
                        x[i, j] = True
            return x

        start = time.time()
        background = masks == 0
        # e.g. a cell within 4 pixels of another one if enlarging both of them by 2 pixels leads to overlap
        p = round(GRAPH_CONTACT_PIXELS / 2)
        neighbors_mask = create_binary_circle(p)
        masks_count_per_pixel = np.zeros_like(background, dtype=np.int16)
        # all_masks = np.zeros_like(background, dtype=np.int16)
        dilated_masks = {}
        bb_coordinates = {}
        for mask_id in mask_ids:
            mask = masks == mask_id
            sum0 = np.cumsum(np.sum(mask, axis=1) > 0)
            sum1 = np.cumsum(np.sum(mask, axis=0) > 0)
            i_a = max(np.argmax(sum0 > 0) - p, 0)
            i_b = min(np.argmax(sum0) + p + 1, background.shape[0])
            j_a = max(np.argmax(sum1 > 0) - p, 0)
            j_b = min(np.argmax(sum1) + p + 1, background.shape[1])
            dilated_bounding_box = np.zeros_like(background)
            bb = np.ones((i_b - i_a, j_b - j_a))
            dilated_bounding_box[i_a:i_b, j_a:j_b] = bb

            # dilated_mask = binary_dilation(mask, neighbors_mask)
            dilated_mask = binary_dilation(
                mask, neighbors_mask, mask=dilated_bounding_box
            )

            def whats_going_on():
                bbb = dilated_mask * 10 + mask  # + dilated_bounding_box * 5
                plt.figure()
                plt.imshow(bbb)
                plt.show()
                with numpy.printoptions(threshold=numpy.inf):
                    open("a.txt", "w").write(
                        str(bbb[:50, :50])
                        .replace("\n", "")
                        .replace(" 0", " .")
                        .replace("]", "]\n")
                    )
                print(neighbors_mask.astype(np.int32))

            # whats_going_on()

            dilated_masks[mask_id] = dilated_mask
            bb_coordinates[mask_id] = [i_a, i_b, j_a, j_b]
            # a = np.logical_and(background, dilated_mask)
            masks_count_per_pixel += dilated_mask
            # all_masks += mask
        print(f"subsetting pixels: {time.time() - start}")

        # # not imporving the perfomance, so I have commented it out
        # from intervaltree import IntervalTree
        # t_i = IntervalTree()
        # t_j = IntervalTree()
        # for mask_index, (i_a, i_b, j_a, j_b) in bb_coordinates.items():
        #     t_i[i_a:i_b] = mask_index
        #     t_j[j_a:j_b] = mask_index
        #
        # def stab(i, j):
        #     stabbed_i = t_i[i]
        #     masks_indexes_i = set(ii.data for ii in stabbed_i)
        #     stabbed_j = t_j[j]
        #     masks_indexes_j = set(jj.data for jj in stabbed_j)
        #     stabbed = masks_indexes_i.intersection(masks_indexes_j)
        #     return stabbed

        start = time.time()
        indices = np.argwhere(masks_count_per_pixel > 1).tolist()
        d = {tuple(index): [] for index in indices}
        for index in d.keys():
            i, j = index
            # stabbed = stab(i, j)
            # for mask_id in stabbed:
            #     dilated_mask = dilated_masks[mask_id]
            for mask_id, dilated_mask in dilated_masks.items():
                if dilated_masks[mask_id][index]:
                    d[index].append(mask_id)
        print(f"determining overlap: {time.time() - start}")

        start = time.time()
        neighbors = {mask_id: set() for mask_id in mask_ids}
        for adjacent in d.values():
            for a in adjacent:
                for b in adjacent:
                    neighbors[a].add(b)
                    neighbors[b].add(a)

        for a, bb in neighbors.items():
            for b in bb:
                # warning: here we subtract 1 because in the region_center numpy array we have already removed
                # the background, while a and b are using the masks indexing, which are contiguous and start
                # from 1 (we are not considering the background)
                edges.append([a - 1, b - 1])
                weights.append(1)
        print(f"building the graph: {time.time() - start}")
        # assert False
        #

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(all_masks > 0)
        # plt.subplot(1, 2, 2)
        # plt.imshow(masks_count_per_pixel > 1)
        # plt.show()
        # assert False
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float).reshape((-1, 1))
    # TODO: ADD EXTRA DATA HERE

    data = Data(
        edge_index=edge_index.long(),
        edge_attr=edge_attr,
        regions_centers=regions_centers,
        num_nodes=len(regions_centers),
    )
    f = get_graph_file(graph_method, split, ome_index)
    torch.save(data, f)


def load_graph(graph_method, split, ome_index):
    validate_graph_method(graph_method)
    f = get_graph_file(graph_method, split, ome_index)
    data = torch.load(f)
    return data


class GraphIMC(InMemoryDataset):
    def __init__(self, split: str, graph_method: str):
        super(GraphIMC, self).__init__()
        validate_graph_method(graph_method)
        self.split = split
        self.graph_method = graph_method
        self.graphs = []
        n = len(FilteredMasksRelabeled(split))
        for i in range(n):
            data = load_graph(graph_method, split, i)
            self.graphs.append(data)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i]


def plot_single_graph(graph_method, split, ome_index):
    data = load_graph(graph_method, split, ome_index)
    plot_imc_graph(data, split, ome_index)
    plot_hist(data)


##
TEST_GRAPH_CREATION = False
# TEST_GRAPH_CREATION = True
if m and TEST_GRAPH_CREATION:
    graph_method = "knn"
    split = "validation"
    ome_index = 1
    compute_graphs(graph_method, split, ome_index)
    plot_single_graph(graph_method, split, ome_index)

##
if m and COMPUTE_GRAPH_FILES:  ## and False and False:
    # graph_method = "gaussian"
    graph_method = "knn"
    for split in tqdm(["validation", "train"], desc="split"):  # , "test"
        n = len(FilteredMasksRelabeled(split=split))
        for ome_index in tqdm(range(n), desc="making graphs"):
            compute_graphs(graph_method, split, ome_index)
            # break
        # break

##
if m and PLOT:
    split = "validation"
    graph_method = "gaussian"
    ds = GraphIMC(split, graph_method)
    for i in [0, 1, 2, 3, 4, 5]:
        plot_single_graph(graph_method, split, i)
        # plot_imc_graph(ds[i], split, i)
        # plot_hist(ds[i])

##

from data2 import IndexInfo


class CellGraph(InMemoryDataset):
    def __init__(self, split: str, graph_method: str):
        self.root = file_path(f"subgraphs_{split}_{graph_method}")
        os.makedirs(self.root, exist_ok=True)
        self.split = split
        validate_graph_method(graph_method)
        self.graph_method = graph_method
        self.graph_imc = GraphIMC(split, graph_method)
        self.ii = IndexInfo(split)
        self.begins = np.array(self.ii.filtered_begins)
        self.ends = np.array(self.ii.filtered_ends)
        self.cells_count = np.sum(self.ii.ok_size_cells)
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["subgraphs.pt"]

    def _compute_and_save(self, cell_index: int):
        subgraph = self.compute_subgraph(cell_index=cell_index)
        subgraph.dump()
        self.pbar.update(1)

    def get_ome_index_from_cell_index(self, cell_index: int):
        i = self.begins.searchsorted(cell_index)
        if i == len(self.begins):
            i -= 1
        elif self.begins[i] != cell_index:
            assert self.begins[i] > cell_index
            i -= 1
        # print(f'cell_index = {cell_index} -> i = {i}')
        # print(f'self.begins[...] = {self.begins[i - 1]}, {self.begins[i]}, {self.begins[i + 1]}')
        # print(f'self.ends[...] = {self.ends[i - 1]}, {self.ends[i]}, {self.ends[i + 1]}')
        # print()
        assert self.begins[i] <= cell_index, (
            self.begins[i - 1],
            self.begins[i],
            self.begins[i + 1],
            cell_index,
        )
        assert cell_index < self.ends[i], (
            self.ends[i - 1],
            self.ends[i],
            self.ends[i + 1],
            cell_index,
        )
        local_cell_index = cell_index - self.begins[i]
        return i, local_cell_index

    def len(self):
        return self.cells_count

    def process(self):
        data_list = []
        for cell_index in tqdm(range(self.cells_count)):
            data = self.compute_subgraph(cell_index)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def compute_subgraph(self, cell_index):
        ome_index, local_cell_index = self.get_ome_index_from_cell_index(cell_index)
        data = self.graph_imc.graphs[ome_index].clone()
        # we could also remove this assertion...
        assert len(data.regions_centers) == data.num_nodes
        center = data.regions_centers[local_cell_index]
        if self.graph_method == "gaussian" or self.graph_method == "contact":
            is_near = (
                np.linalg.norm(data.regions_centers - center, axis=1) < SUBGRAPH_RADIUS
            )
        elif self.graph_method == "knn":
            g = to_networkx(data, edge_attrs=["edge_attr"], node_attrs=["regions_centers"])
            neighbors = list(g.neighbors(local_cell_index))
            l = []
            for node in neighbors:
                w = g.get_edge_data(local_cell_index, node)['edge_attr']
                l.append((node, w))
            l = sorted(l, key=lambda x: x[1])
            nearest = [l[i][0] for i in range(GRAPH_KNN_SUBGRAPH_K - 1)]
            nearest.append(local_cell_index)
            is_near = np.zeros(len(data.regions_centers), dtype=np.bool)
            is_near[np.array(nearest, dtype=np.long)] = True
        else:
            raise ValueError()

        nodes_to_keep = torch.arange(data.num_nodes, dtype=torch.long)[is_near]
        sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
            nodes_to_keep,
            data.edge_index,
            edge_attr=data.edge_attr,
            relabel_nodes=True,
        )
        center_index = (np.cumsum(is_near) - 1)[local_cell_index]
        sub_data = Data(
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            regions_centers=data.regions_centers[is_near],
            center_index=center_index,
            is_near=is_near,
        )
        sub_data.num_nodes = len(sub_data.regions_centers)
        return sub_data


def plot_single_cell_graph(
    cell_graph: CellGraph, cell_index: int, plot_expression: bool = False
):
    data = cell_graph[cell_index]
    ome_index, _ = cell_graph.get_ome_index_from_cell_index(cell_index)
    plot_imc_graph(
        data, cell_graph.split, ome_index=ome_index, plot_expression=plot_expression
    )


if m and COMPUTE_SUBGRAPH_DATASET:
    ds = CellGraph(split="validation", graph_method="knn")
    plot_single_cell_graph(cell_graph=ds, cell_index=999)
    # CellGraph(split="validation", graph_method="gaussian")
    # CellGraph(split="train", graph_method="gaussian")
    # CellGraph(split='test', graph_method='gaussian')

if m and PLOT:
    # ds = CellGraph("validation", "gaussian")
    ds = CellGraph("validation", "knn")
    print(ds[0])
    plot_single_cell_graph(cell_graph=ds, cell_index=999)

# TODO: alskdjasjdl

# import sys
# sys.exit(0)
##
# if m:
# ds = CellGraph("validation", "gaussian")
# a0, a1, a2, a3 = ds.begins[0], ds.begins[1] - 1, ds.begins[1], ds.begins[1] + 1
# x0, x1, x2, x3 = ds[a0], ds[a1], ds[a2], ds[a3]
# plot_single_cell_graph(cell_graph=ds, cell_index=a0)
# plot_single_cell_graph(cell_graph=ds, cell_index=a1)
# plot_single_cell_graph(cell_graph=ds, cell_index=a2)
# plot_single_cell_graph(cell_graph=ds, cell_index=a3)
# plot_single_cell_graph(cell_graph=ds, cell_index=22190)


##
import torch.utils.data


class CellExpressionGraph(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        graph_method: str,
        perturb: bool = False,
        perturb_entire_cells: bool = False,
    ):
        super().__init__()
        self.split = split
        self.graph_method = graph_method
        self.cell_graph = CellGraph(split=split, graph_method=graph_method)
        self.cell_ds = PerturbedCellDataset(split=split)
        if perturb_entire_cells:
            self.cell_ds.perturb_entire_cells()
        elif perturb:
            self.cell_ds.perturb()
        # needed from a plotting function, so I can call it also on this class
        self.get_ome_index_from_cell_index = (
            self.cell_graph.get_ome_index_from_cell_index
        )
        assert len(self.cell_graph) == len(self.cell_ds)
        # to speed up __getitem__
        self.merged_expressions = None
        self.merged_is_perturbed = None

    def merge(self):
        expressions, are_perturbed = merge_perturbed_cell_dataset(self.cell_ds)
        self.merged_expressions = torch.from_numpy(expressions)
        self.merged_is_perturbed = torch.from_numpy(are_perturbed)

    def __len__(self):
        return len(self.cell_graph)

    def __getitem__(self, i):
        data = self.cell_graph[i]
        l_expression = []
        l_is_perturbed = []
        ome_index, local_cell_index = self.cell_graph.get_ome_index_from_cell_index(
            cell_index=i
        )
        begin = self.cell_graph.ii.filtered_begins[ome_index]
        assert begin + local_cell_index == i
        indices = np.arange(len(data.is_near))[data.is_near] + begin
        for ii in indices:
            # we have that self.merged_expressions is None exactly when self.merged_is_perturbed is
            if self.merged_expressions is None:
                expression, _, is_perturbed = self.cell_ds[ii]
            else:
                expression = self.merged_expressions[ii, :]
                is_perturbed = self.merged_is_perturbed[ii, :]
            l_expression.append(expression.reshape(1, -1))
            l_is_perturbed.append(is_perturbed.reshape(1, -1))
        expressions = np.concatenate(l_expression, axis=0)
        are_perturbed = np.concatenate(l_is_perturbed, axis=0)
        data.x = torch.from_numpy(expressions)
        data.is_perturbed = are_perturbed
        return data


#
# ds0 = CellExpressionGraph(split='validation', graph_method='gaussian', perturb=True)
# ds1 = CellExpressionGraph(split='validation', graph_method='gaussian', perturb=True)
# ds1.merge()
#
# for i in tqdm(range(len(ds0))):
#     data0 = ds0[i]
# data1 = ds1[i]
# assert np.array_equal(data0.is_perturbed, data1.is_perturbed)
#
# import torch_geometric.data
# loader0 = torch_geometric.data.DataLoader(ds0, batch_size=32, num_workers=8, pin_memory=True)
# loader1 = torch_geometric.data.DataLoader(ds1, batch_size=32, num_workers=8, pin_memory=True)
# for data0, data1 in tqdm(zip(loader0, loader1), total=len(loader0)):
#     x0 = np.concatenate(data0.is_perturbed, axis=0)
#     x1 = np.concatenate(data1.is_perturbed, axis=0)
#     assert np.all(x0 == x1)
##

if m and PLOT:
    ds = CellExpressionGraph(split="validation", graph_method="gaussian")
    x = ds[0]
    print(x.x.shape, x.num_nodes)
    plot_single_cell_graph(cell_graph=ds, cell_index=999, plot_expression=True)
    plot_single_cell_graph(cell_graph=ds, cell_index=100000, plot_expression=True)

##
import torch.utils.data


class CellExpressionGraphOptimized(InMemoryDataset):
    def __init__(
        self,
        split: str,
        graph_method: str,
        perturb: bool = False,
        perturb_entire_cells: bool = False,
    ):
        if perturb_entire_cells:
            p = "_perturbed_cells"
        elif perturb:
            p = "_perturbed"
        else:
            p = ""
        self.root = file_path(f"subgraphs_expression_{split}_{graph_method}{p}")
        os.makedirs(self.root, exist_ok=True)
        self.split = split
        validate_graph_method(graph_method)
        self.graph_method = graph_method
        self.cell_expression_graph = CellExpressionGraph(
            split,
            graph_method,
            perturb=perturb,
            perturb_entire_cells=perturb_entire_cells,
        )
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["subgraphs.pt"]

    def len(self):
        return self.cell_expression_graph.cell_graph.cells_count

    def process(self):
        data_list = []
        self.cell_expression_graph.merge()
        for cell_index in tqdm(range(len(self.cell_expression_graph))):
            data = self.cell_expression_graph[cell_index]
            data.is_center = torch.zeros(len(data.x), dtype=torch.float)
            data.is_center[data.center_index] = 1.0
            del data.center_index
            del data.is_near
            del data.regions_centers
            data.is_perturbed = torch.from_numpy(data.is_perturbed)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# #
# import gc
#
# gc.collect()
# #
# ds2 = CellExpressionGraphOptimized('validation', 'gaussian', perturb=True)
# ##
# loader2 = torch_geometric.data.DataLoader(ds2, batch_size=32, num_workers=0, pin_memory=True)
# for data0, data2 in tqdm(zip(loader0, loader2), total=len(loader0)):
#     is_perturbed0 = data0.is_perturbed
#     is_perturbed2 = data2.is_perturbed[torch.where(data2.is_center == 1.)[0], :]
#     x0 = np.concatenate(is_perturbed0, axis=0)
#     x00 = x0[torch.where(data2.is_center == 1.)[0].numpy(), :]
#     x2 = is_perturbed2.numpy()
#     assert np.all(x00 == x2)

##
if m and COMPUTE_OPTIMIZED_SUBGRAPH_DATASET:
    CellExpressionGraphOptimized("validation", "gaussian")
    # CellExpressionGraphOptimized('train', 'gaussian')
    # CellExpressionGraphOptimized('test', 'gaussian')

##
if m and TEST:
    ds0 = CellGraph("validation", "gaussian")
    ds1 = CellExpressionGraphOptimized("validation", "gaussian")
    for i in tqdm(range(500)):
        x = ds0[i]
    for i in tqdm(range(500)):
        y = ds1[i]

##
if m and TEST:
    from torch_geometric.data import DataLoader as GeometricDataLoader

    ds = CellExpressionGraphOptimized("train", "gaussian")
    ##
    loader = GeometricDataLoader(
        ds,
        batch_size=32,
        num_workers=16,
        pin_memory=True,
        shuffle=True,
    )
    x = loader.__iter__().__next__()
    print(x)

##
if m and COMPUTE_PERTURBED_DATASET:
    CellExpressionGraphOptimized("validation", "gaussian", perturb=True)
# class CellExpressionGraphOptimizedPerturbed(torch_geometric.data.Dataset):
#     def __init__(self, split: str, graph_method: str):
#         super().__init__()
#         self.ds = CellExpressionGraphOptimized(split, graph_method)
#         self.perturbed_cells_ds = PerturbedCellDataset(split=split)
#         self.perturbed_cells_ds.perturb()
#         self.is_perturbed = self.is_perturbed
#
#     def __len__(self):
#         return len(self.ds)
#
#     def __getitem__(self, i):
#         data = self.ds[i]

##
"""
FULL PERTURBATION
"""
if m and COMPUTE_PERTURBED_CELLS_DATASET:
    ds = CellExpressionGraphOptimized(
        split="validation", graph_method="gaussian", perturb_entire_cells=True
    )
