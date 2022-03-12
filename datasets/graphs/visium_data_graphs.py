##
import math
import os
import shutil
import time

import colorama
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import spatialmuon
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm.auto import tqdm

from datasets.visium_data import get_smu_file, get_split_indices
from utils import get_execute_function, file_path, get_bimap, reproducible_random_choice

e_ = get_execute_function()

plt.style.use("dark_background")

##
if e_():
    print(f"{colorama.Fore.MAGENTA}computing graphs{colorama.Fore.RESET}")
    s = get_smu_file(read_only=False)
    g = s["visium"]["processed"].compute_proximity_graph(max_distance_in_units=200)

    _, ax = plt.subplots(1)
    g.plot(node_colors="white", edge_colors="white", ax=ax)
    ax.set(xlim=(1000, 2000), ylim=(2500, 4000))
    plt.show()

    if "graph" in s["visium"]["processed"]:
        del s["visium"]["processed"]["graph"]
    s["visium"]["processed"].graph = g
##
if e_():
    _, ax = plt.subplots(figsize=(10, 10))
    s["visium"]["image"].plot(ax=ax)
    bb = spatialmuon.BoundingBox(x0=0, x1=2000, y0=2500, y1=4500)
    s["visium"]["processed"].graph.plot(
        node_colors=None,
        node_size=1,
        edge_colors=[0.0, 0.0, 0.0, 1],
        edge_size=0.5,
        ax=ax,
    )
    s["visium"]["processed"].plot(0, ax=ax, bounding_box=bb)
    bb.set_lim_for_ax(ax)
    plt.show()
##
if e_():
    _, ax = plt.subplots(figsize=(10, 10))
    s["visium"]["image"].plot(ax=ax)
    s["visium"]["processed"].plot(0, fill_color=None, outline_color="channel", ax=ax)
    s["visium"]["processed"].graph.plot(
        node_colors="w",
        node_size=1,
        edge_colors=[0.0, 0.0, 0.0, 0.3],
        edge_size=1,
        ax=ax,
    )
    ax.set(xlim=(2200, 2900), ylim=(6000, 6700))
    plt.show()
##
class CellGraphsDataset(InMemoryDataset):
    def __init__(self, split: str, name: str):
        os.makedirs(file_path("visium_mousebrain/"), exist_ok=True)
        self.root = file_path(f"visium_mousebrain/subgraphs_{split}_{name}")
        os.makedirs(self.root, exist_ok=True)
        self.split = split
        self.name = name
        self.s = get_smu_file(read_only=True)
        self.split_indices = get_split_indices(split)

        start = time.time()
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"loading from disk: {time.time() - start}")
        self.s.backing.close()

    @property
    def processed_file_names(self):
        return ["subgraphs.pt"]

    def len(self):
        return len(self.split_indices)

    def process(self):
        data_list = []
        g = self.s["visium"]["processed"].graph
        n = len(self.s["visium"]["processed"].X)
        x = self.s["visium"]["processed"].X[...]
        (
            list_of_sub_g,
            list_of_center_index,
            list_of_original_index,
        ) = g.subgraph_of_neighbors(
            node_indices=self.split_indices, subset_method="knn"
        )
        for sub_g, center_index, original_indices in zip(
            list_of_sub_g, list_of_center_index, list_of_original_index
        ):
            e = x[original_indices, :]
            sub_data = Data(
                edge_index=sub_g.edge_indices.T,
                edge_attr=sub_g.edge_features,
                pos=sub_g.untransformed_node_positions,
                x=e,
                center_index=center_index,
                original_indices=original_indices,
            )
            sub_data.num_nodes = n
            data_list.append(sub_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


##
if e_():
    CURRENT_SUBGRAPH_NAME = "contact_200"

if e_():
    f = file_path(
        f"visium_mousebrain/subgraphs_train_{CURRENT_SUBGRAPH_NAME}/processed"
    )
    if os.path.isdir(f):
        shutil.rmtree(f)

##
if e_():
    f = file_path(
        f"visium_mousebrain/subgraphs_validation_{CURRENT_SUBGRAPH_NAME}/processed"
    )
    if os.path.isdir(f):
        shutil.rmtree(f)

##
if e_():
    f = file_path(f"visium_mousebrain/subgraphs_test_{CURRENT_SUBGRAPH_NAME}/processed")
    if os.path.isdir(f):
        shutil.rmtree(f)

##
if e_():
    ds = CellGraphsDataset(split="train", name=CURRENT_SUBGRAPH_NAME)

if e_():
    ds = CellGraphsDataset(split="validation", name=CURRENT_SUBGRAPH_NAME)

if e_():
    ds = CellGraphsDataset(split="test", name=CURRENT_SUBGRAPH_NAME)
    print(ds[0])

##
if e_():
    data = ds[0]
    s = get_smu_file(read_only=True)
    g = spatialmuon.Graph(
        untransformed_node_positions=data.pos,
        edge_indices=data.edge_index.T,
        edge_features=data.edge_attr,
        undirected=data.is_undirected,
    )
    r = spatialmuon.Regions(graph=g, anchor=s["visium"]["processed"].anchor)
    _, ax = plt.subplots(1, figsize=(10, 10))
    s["visium"]["image"].plot(ax=ax)
    s["visium"]["processed"].plot(0, ax=ax, alpha=0.2)
    r.graph.plot(node_colors="k", edge_colors="k", ax=ax)
    ax.set(xlim=(1000, 1750), ylim=(4250, 5000))
    original_center_index = data.original_indices[data.center_index]
    xy = s["visium"]["processed"].transformed_centers[original_center_index]
    ax.scatter(xy[0], xy[1], c="g")
    plt.show()
