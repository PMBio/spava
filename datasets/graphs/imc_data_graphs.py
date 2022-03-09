##
import spatialmuon as smu

import random
import math

import shutil
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import time
import h5py
import pickle
import skimage
import skimage.io
import torch
import cv2
import torchvision.transforms
import vigra
import os

from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import splits
import matplotlib
import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import DataLoader  # SequentialSampler
import sys
import tempfile

import pathlib
from utils import setup_ci, file_path, get_bimap
import colorama
from datasets.imc_data import get_smu_file, get_split, all_processed_smu

# os.environ['SPATIALMUON_TEST'] = 'aaa'
# os.environ['SPATIALMUON_NOTEBOOK'] = 'aaa'
c_, p_, t_, n_ = setup_ci(__name__)
# matplotlib.use('module://backend_interagg')

plt.style.use("dark_background")

##
# if n_ or t_ or c_:
if n_ or t_ or c_ and False:
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
# if n_ or t_ or p_:
if n_ or t_ or p_ and False:
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
# if n_ or t_ or p_:
if n_ or t_ or p_ and False:
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
        self.root = file_path(f"subgraphs_{split}_{name}")
        os.makedirs(self.root, exist_ok=True)
        self.split = split
        self.name = name
        self.filenames = get_split(self.split)
        names_length_map = {}
        for i in range(len(self.filenames)):
            s = get_smu_file(self.split, i)
            n = len(s["imc"]["transformed_mean"].X)
            names_length_map[i] = n
        self.map_left, self.map_right = get_bimap(names_length_map)

        start = time.time()
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f'loading from disk: {time.time() - start}')

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
            s = get_smu_file(self.split, i)
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
                data_list.append(sub_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
##
# if n_ or t_ or p_:
if n_ or t_ or p_ and False:
    CURRENT_SUBGRAPH_NAME = "knn_10_max_distance_in_units_50"

# if n_ or t_ or p_:
if n_ or t_ or p_ and False:
    f = file_path(f"subgraphs_train_{CURRENT_SUBGRAPH_NAME}/processed")
    if os.path.isdir(f):
        shutil.rmtree(f)
##
# if n_ or t_ or p_:
if n_ or t_ or p_ and False:
    f = file_path(f"subgraphs_validation_{CURRENT_SUBGRAPH_NAME}/processed")
    if os.path.isdir(f):
        shutil.rmtree(f)
##
# if n_ or t_ or p_:
if n_ or t_ or p_ and False:
    f = file_path(f"subgraphs_test_{CURRENT_SUBGRAPH_NAME}/processed")
    if os.path.isdir(f):
        shutil.rmtree(f)
##
# if n_ or t_ or p_:
if n_ or t_ or p_ and False:
    ds = CellGraphsDataset(
        split="train", name=CURRENT_SUBGRAPH_NAME
    )
    ds = CellGraphsDataset(
        split="validation", name=CURRENT_SUBGRAPH_NAME
    )
    ds = CellGraphsDataset(
        split="test", name=CURRENT_SUBGRAPH_NAME
    )
##
