import torch

# from torch.functional import F
from torch_geometric.data import Data, InMemoryDataset
import networkx
import numpy as np
import numpy.linalg
from tqdm import tqdm
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import math
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils.convert import to_networkx

# from skimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_dilation
import random
import colorsys
import h5py
import time
from pprint import pprint
from data2 import (
    AccumulatedDataset,
    ExpressionFilteredDataset,
    FilteredMasksRelabeled,
    file_path,
    quantiles_for_normalization,
)
from torch.utils.data import Dataset
import os


class PerturbedCellDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.ds = AccumulatedDataset(
            split, feature="region_center", from_raw=False, transform=False
        )
        self.index_converter = FilteredMasksRelabeled(
            split
        ).get_indices_conversion_arrays
        all = []
        for i in tqdm(range(len(self.ds)), desc="merging expression tensor"):
            e = self.ds[i]
            new_e = ExpressionFilteredDataset.expression_old_to_new(
                e, i, index_converter=self.index_converter
            )
            all.append(new_e)

    def __len__(self):
        return len(self.merged)

    def __getitem__(self, i):
        return self.merged[i, :], self.corrupted_entries[i, :]


def plot_imc_graph(data, ome_index=None, custom_ax=None):
    g = to_networkx(data, edge_attrs=["edge_attr"], node_attrs=["regions_centers"])
    positions = {i: d["regions_centers"] for i, d in g.nodes(data=True)}
    weights = np.array([x[2]["edge_attr"] for x in g.edges(data=True)])
    edges_to_plot = [(e[0], e[1]) for e in g.edges(data=True) if e[2]["edge_attr"] > 0]
    # f = networkx.Graph()
    # f_edges = filter(lambda x: x[2]['edge_attr'] > 0, g.edges(data=True))
    # f.add_edges_from(f_edges)
    if custom_ax is None:
        plt.figure(figsize=(16, 9))
        ax = plt.gca()
    else:
        ax = custom_ax
    ax.set_facecolor((1.0, 0.47, 0.42))
    greys = matplotlib.cm.get_cmap("Greys")

    if ome_index is not None:
        ome_filenames = get_ome_filenames()
        colors = [
            colorsys.hsv_to_rgb(5 / 360, 58 / 100, (60 + random.random() * 40) / 100)
            for _ in range(10000)
        ]
        colors[0] = colorsys.hsv_to_rgb(5 / 360, 58 / 100, 63 / 100)
        new_map = matplotlib.colors.LinearSegmentedColormap.from_list(
            "new_map", colors, N=10000
        )

        with h5py.File(configs_uzh.paths.relabelled_masks_file, "r") as f5:
            masks = f5[ome_filenames[ome_index] + "/masks"][...]
        ax.imshow(np.moveaxis(masks, 0, 1), cmap=new_map, aspect="auto")
        # ax = plt.gca()
    # else:
    #     ax = None

    networkx.drawing.nx_pylab.draw_networkx(
        g,
        pos=positions,
        edgelist=edges_to_plot,
        node_size=10,
        with_labels=False,
        linewidths=0.5,
        arrows=False,
        node_color="#ff0000",
        edge_color=weights,
        edge_cmap=greys,
        edge_vmin=np.min(weights),
        edge_vmax=np.max(weights),
        ax=ax,
    )

    if custom_ax is None:
        plt.show()
    # return ax


def plot_hist(data):
    x = data.edge_attr.numpy()
    plt.figure()
    plt.hist(x)
    plt.show()


def compute_graphs(instance: Instance):
    assert instance.graph_method in ["gaussian", "knn", "contact", None]
    print(f"instance.graph_method = {instance.graph_method}")
    ome_filenames = get_ome_filenames()
    for ome_index, ome_filename in enumerate(
        tqdm(ome_filenames, desc="building torch geometric graphs")
    ):
        # if ome_index > 0:
        #     return
        with h5py.File(configs_uzh.paths.region_centers_file, "r") as f5:
            regions_centers = f5[ome_filename + "/region_center"][...]
        edges = []
        weights = []

        if instance.graph_method is None:
            # to make snakemake happy
            f = configs_uzh.paths.get_graph_file(instance, ome_index)
            open(f, "w")
            return
        elif instance.graph_method == "gaussian":
            tree = cKDTree(regions_centers)
            for i in range(len(regions_centers)):
                a = np.array(regions_centers[i])
                r_threshold = instance.graph_gaussian_r_threshold
                # in order to be in the graph, two cells must have their center closer than d_threshold microns (i.e.
                # d_threshold pixels in the image)
                l = instance.graph_gaussian_l
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
        elif instance.graph_method == "knn":
            k = instance.graph_knn_k
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
        elif instance.graph_method == "contact":
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
            with h5py.File(configs_uzh.paths.relabelled_masks_file, "r") as f5:
                masks = f5[ome_filename + "/masks"][...]
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
            p = round(instance.graph_contact_pixels / 2)
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
        f = configs_uzh.paths.get_graph_file(instance, ome_index)
        torch.save(data, f)


class GraphIMC(InMemoryDataset):
    def __init__(self, instance: Instance):
        super(GraphIMC, self).__init__()
        self.instance = instance
        self.ome_filenames = get_ome_filenames()
        self.graphs = []
        for i, ome_filename in enumerate(self.ome_filenames):
            f = configs_uzh.paths.get_graph_file(instance, i)
            data = torch.load(f)
            self.graphs.append(data)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i]


# it takes more time but less code to create this object from a CellLevelPhenographClustering object
# anyway, once we precompute that object, the execution time is fast


if __name__ == "__main__":
    from spatial_uzh.configs.models import instances, Resource

    pprint(
        instances[0].get_df_of_instances(
            instances=instances, resource_name=Resource.graphs
        )
    )
    # compute_graphs(instances[0])
    for instance in instances:
        if instance.graph_method is None:
            continue
        print(instance.graph_method)
        ds = GraphIMC(instance)
        for i in [0]:
            plot_imc_graph(ds[i], ome_index=i)
            plot_hist(ds[i])
