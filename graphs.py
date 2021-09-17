import colorsys
import math
import os
import random
import time
import dill
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx
import numpy as np
import numpy.linalg
import torch
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

GRAPH_CONTACT_PIXELS = 4
GRAPH_GAUSSIAN_R_THRESHOLD = 0.1
GRAPH_GAUSSIAN_L = 400
GRAPH_KNN_K = 10
SUBGRAPH_RADIUS = 75

m = __name__ == "__main__"


def plot_imc_graph(data, split: str = None, ome_index=None, custom_ax=None):
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
    ax.set_facecolor((1.0, 0.47, 0.42))
    greys = matplotlib.cm.get_cmap("Greys_r")

    if ome_index is not None:
        assert split is not None
        masks_ds = FilteredMasksRelabeled(split)
        colors = [
            colorsys.hsv_to_rgb(5 / 360, 58 / 100, (60 + random.random() * 40) / 100)
            for _ in range(10000)
        ]
        colors[0] = colorsys.hsv_to_rgb(5 / 360, 58 / 100, 63 / 100)
        new_map = matplotlib.colors.LinearSegmentedColormap.from_list(
            "new_map", colors, N=10000
        )

        masks = masks_ds[ome_index]
        ax.imshow(np.moveaxis(masks, 0, 1), cmap=new_map, aspect="auto")
        # ax = plt.gca()
    # else:
    #     ax = None
    node_colors = ["#ff0000"] * len(positions)
    if hasattr(data, "center_index"):
        node_colors[data.center_index] = "#00ff00"

    networkx.drawing.nx_pylab.draw_networkx(
        g,
        pos=positions,
        edgelist=edges_to_plot,
        node_size=10,
        with_labels=False,
        linewidths=0.5,
        arrows=False,
        node_color=node_colors,
        edge_color=weights,
        edge_cmap=greys,
        edge_vmin=np.min(weights),
        edge_vmax=np.max(weights),
        ax=ax,
    )

    if custom_ax is None:
        plt.show()
    # return ax

    # # -------------
    # plot_hist(data)
    # ##


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
        # check that every element is a knn of itself, and this is the first neighbor
        assert all(indices[:, 0] == np.array(range(len(regions_centers))))
        edges = []
        weights = []
        for i in range(indices.shape[0]):
            for j in range(1, indices.shape[1]):
                edge = [i, indices[i, j]]
                weight = 1
                # weight = 1 / (distances[i, j] ** 2 + 1)
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

    data = Data(
        edge_index=edge_index,
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
if m and False and False:
    graph_method = "gaussian"
    for split in tqdm(["validation", "train", "test"], desc="split"):
        n = len(FilteredMasksRelabeled(split=split))
        for ome_index in tqdm(range(n), desc="making graphs"):
            compute_graphs(graph_method, split, ome_index)
            # break
        # break

##
if m and False:
    split = "validation"
    graph_method = "gaussian"
    ds = GraphIMC(split, graph_method)
    for i in [0, 1, 2, 3, 4, 5]:
        plot_single_graph(graph_method, split, i)
        # plot_imc_graph(ds[i], split, i)
        # plot_hist(ds[i])

##

from data2 import IndexInfo


class Subgraph:
    def __init__(self, split, data, cell_index, ome_index, local_cell_index, relabeler):
        self.split = split
        self.data = data
        self.cell_index = cell_index
        self.ome_index = ome_index
        self.local_cell_index = local_cell_index
        self.relabeler = relabeler

    def dump(self):
        f = os.path.join("subgraphs", f"{self.split}_{self.cell_index}.pickle")
        d = {
            "split": self.split,
            "data": self.data,
            "cell_index": self.cell_index,
            "ome_index": self.ome_index,
            "local_cell_index": self.local_cell_index,
            "relabeler": self.relabeler,
        }
        dill.dump(d, open(f, "wb"))

    @classmethod
    def load(cls, split, cell_index):
        f = os.path.join("subgraphs", f"{split}_{cell_index}.pickle")
        d = dill.load(open(f, "rb"))
        subgraph = Subgraph(
            split=d["split"],
            data=d["data"],
            cell_index=d["cell_index"],
            ome_index=d["ome_index"],
            local_cell_index=d["local_cell_index"],
            relabeler=d["relabeler"],
        )
        return subgraph


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
        # self.cells_count = np.sum(self.ii.ok_size_cells)
        self.cells_count = 1000
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # # if False:
        # if True:
        #     f = file_path("subgraphs")
        #     os.makedirs(f, exist_ok=True)
        #     # i = 0
        #     # while os.path.exists(file_path(f"subgraphs{i}")):
        #     #     i += 1
        #     # f = file_path(f"subgraphs{i}")
        #     from multiprocessing import Pool
        #
        #     self.pbar = tqdm(total=self.cells_count)
        #
        #     with Pool(4) as p:
        #         p.map(self._compute_and_save, list(range(self.cells_count)))
        #     self.pbar.close()

    @property
    def processed_file_names(self):
        return ["subgraphs.pt"]

    def _compute_and_save(self, cell_index: int):
        subgraph = self.compute_subgraph(cell_index=cell_index)
        subgraph.dump()
        self.pbar.update(1)

    def get_ome_index_from_cell_index(self, cell_index: int):
        i = self.begins.searchsorted(cell_index)
        if self.begins[i] != cell_index:
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
        is_near = (
            np.linalg.norm(data.regions_centers - center, axis=1) < SUBGRAPH_RADIUS
        )
        to_keep = set(np.arange(data.num_nodes)[is_near].tolist())
        edges_to_keep = []
        for i, (a, b) in enumerate(data.edge_index.T.numpy()):
            k = a in to_keep and b in to_keep
            if k:
                edges_to_keep.append(i)
        edges_to_keep = np.array(edges_to_keep)
        data.edge_index = data.edge_index[:, edges_to_keep]
        data.edge_attr = data.edge_attr[edges_to_keep, :]
        relabeler = np.cumsum(is_near)
        # for debugging purposes, see the assert below
        relabeler[np.logical_not(is_near)] = 0
        relabeler -= 1
        data.regions_centers = data.regions_centers[is_near]
        data.edge_index = relabeler[data.edge_index]
        # the min must be 0, not -1
        assert data.edge_index.min() == 0
        data.num_nodes = len(data.regions_centers)
        data.edge_index = torch.tensor(data.edge_index)
        data.regions_centers = torch.tensor(data.regions_centers)
        data.center_index = relabeler[local_cell_index.item()]
        # subgraph = Subgraph(
        #     split=self.split,
        #     data=data,
        #     cell_index=cell_index,
        #     ome_index=ome_index,
        #     local_cell_index=local_cell_index,
        #     relabeler=relabeler,
        # )
        # return subgraph
        return data

    # def get(self, i):
    #     data = torch.load(os.path.join(self.root, "data_{}.pt".format(i)))
    #     return data

    # def __getitem__(self, item):
    #     raise NotImplementedError()

    #     subgraph = Subgraph.load(split=self.split, cell_index=item)
    #     return subgraph.data


def plot_single_cell_graph(cell_graph: CellGraph, cell_index: int):
    data = cell_graph[cell_index]
    ome_index, _ = cell_graph.get_ome_index_from_cell_index(cell_index)
    plot_imc_graph(data, cell_graph.split, ome_index=ome_index)

ds = CellGraph('validation', 'gaussian')
print(ds[0])
import sys
sys.exit(0)
##
if m:
    ds = CellGraph("validation", "gaussian")
    a0, a1, a2, a3 = ds.begins[0], ds.begins[1] - 1, ds.begins[1], ds.begins[1] + 1
    x0, x1, x2, x3 = ds[a0], ds[a1], ds[a2], ds[a3]
    plot_single_cell_graph(cell_graph=ds, cell_index=a0)
    plot_single_cell_graph(cell_graph=ds, cell_index=a1)
    plot_single_cell_graph(cell_graph=ds, cell_index=a2)
    plot_single_cell_graph(cell_graph=ds, cell_index=a3)
    plot_single_cell_graph(cell_graph=ds, cell_index=22190)


##
class CellExpressionGraph(InMemoryDataset):
    def __init__(self, split: str, graph_method: str):
        super().__init__()
        self.split = split
        self.graph_method = graph_method
        self.cell_graph = CellGraph(split=split, graph_method=graph_method)
        self.cell_ds = PerturbedCellDataset(split=split)
        assert len(self.cell_graph) == len(self.cell_ds)

    def __len__(self):
        return len(self.cell_ds)

    def __getitem__(self, i):
        data = self.cell_graph[i]
        expression, _, is_perturbed = self.cell_ds[i]
        data.x = expression
        data.is_perturbed = is_perturbed
        return data


if m:
    ds = CellExpressionGraph(split="validation", graph_method="gaussian")
    x = ds[0]
    print(x.x.shape, x.num_nodes)
