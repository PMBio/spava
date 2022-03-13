##
import os
import shutil
import time

import colorama
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm.auto import tqdm

from datasets.imc_data import get_smu_file, get_split, all_processed_smu
from utils import get_execute_function, file_path, get_bimap

# os.environ['SPATIALMUON_TEST'] = 'aaa'
# os.environ['SPATIALMUON_NOTEBOOK'] = 'aaa'
e_ = get_execute_function()
# matplotlib.use('module://backend_interagg')

plt.style.use("dark_background")

##
if e_():
    print(f"{colorama.Fore.MAGENTA}computing graphs{colorama.Fore.RESET}")
    for s in all_processed_smu():
        g = s["imc"]["transformed_mean"].compute_knn_graph(
            k=10, max_distance_in_units=50
        )
        # g.plot(node_colors='white', edge_colors='white')
        if "graph" in s["imc"]["transformed_mean"]:
            del s["imc"]["transformed_mean"]["graph"]
        s["imc"]["transformed_mean"].graph = g
##

if e_():
    _, ax = plt.subplots(figsize=(10, 10))
    s["imc"]["ome"].plot(
        0, ax=ax, preprocessing=np.arcsinh, cmap=matplotlib.cm.get_cmap("gray")
    )
    s["imc"]["transformed_mean"].plot(0, ax=ax)
    s["imc"]["transformed_mean"].graph.plot(
        node_colors="w",
        node_size=1,
        edge_colors=[1.0, 1.0, 1.0, 0.5],
        edge_size=0.5,
        ax=ax,
    )
    plt.show()
##

if e_():
    _, ax = plt.subplots(figsize=(10, 10))
    s["imc"]["ome"].plot(
        0, ax=ax, preprocessing=np.arcsinh, cmap=matplotlib.cm.get_cmap("gray")
    )
    s["imc"]["transformed_mean"].plot(
        0, fill_color=None, outline_color="channel", ax=ax
    )
    s["imc"]["transformed_mean"].graph.plot(
        node_colors="w",
        node_size=1,
        edge_colors=[1.0, 1.0, 1.0, 0.5],
        edge_size=0.5,
        ax=ax,
    )
    ax.set(xlim=(1, 100), ylim=(1, 100))
    plt.show()
##
class CellGraphsDataset(InMemoryDataset):
    def __init__(self, split: str, name: str):
        os.makedirs(file_path("imc/"), exist_ok=True)
        self.root = file_path(f"imc/subgraphs_{split}_{name}")
        os.makedirs(self.root, exist_ok=True)
        self.split = split
        self.name = name
        self.filenames = get_split(self.split)
        names_length_map = {}
        for i in range(len(self.filenames)):
            s = get_smu_file(split=self.split, index=i, read_only=True)
            n = len(s["imc"]["transformed_mean"].X)
            names_length_map[i] = n
        self.map_left, self.map_right = get_bimap(names_length_map)

        start = time.time()
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"loading from disk: {time.time() - start}")

    @property
    def processed_file_names(self):
        return ["subgraphs.pt"]

    def len(self):
        return len(self.map_left)

    def process(self):
        data_list = []
        for i in tqdm(
            range(len(self.filenames)),
            desc="computing subgraph",
            leave=True,
            position=0,
        ):
            s = get_smu_file(split=self.split, index=i, read_only=True)
            g = s["imc"]["transformed_mean"].graph
            n = len(s["imc"]["transformed_mean"].X)
            x = s["imc"]["transformed_mean"].X[...]
            (
                list_of_sub_g,
                list_of_center_index,
                list_of_original_index,
            ) = g.subgraph_of_neighbors(
                node_indices=list(range(n)), subset_method="knn"
            )
            # for j in tqdm(range(n), 'subgraphs in slide', leave=True, position=0):
            #     sub_g, center_index, original_indices = g.subgraph_of_neighbors(
            #         node_index=j, subset_method="knn"
            #     )
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
    CURRENT_SUBGRAPH_NAME = "knn_10_max_distance_in_units_50"


if e_():
    f = file_path(f"imc/subgraphs_train_{CURRENT_SUBGRAPH_NAME}/processed")
    if os.path.isdir(f):
        shutil.rmtree(f)
##

if e_():
    f = file_path(f"imc/subgraphs_validation_{CURRENT_SUBGRAPH_NAME}/processed")
    if os.path.isdir(f):
        shutil.rmtree(f)
##

if e_():
    f = file_path(f"imc/subgraphs_test_{CURRENT_SUBGRAPH_NAME}/processed")
    if os.path.isdir(f):
        shutil.rmtree(f)
##

if e_():
    ds = CellGraphsDataset(split="train", name=CURRENT_SUBGRAPH_NAME)

if e_():
    ds = CellGraphsDataset(split="validation", name=CURRENT_SUBGRAPH_NAME)

if e_():
    ds = CellGraphsDataset(split="test", name=CURRENT_SUBGRAPH_NAME)
##
