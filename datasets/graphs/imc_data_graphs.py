##
import spatialmuon as smu

import random
import math

import shutil
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
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
if n_ or t_ or c_:
    print(f'{colorama.Fore.MAGENTA}extracting tiles{colorama.Fore.RESET}')
    for s in all_processed_smu():
        del s['imc']['transformed_mean']['graph']
        # g = s['imc']['transformed_mean'].compute_knn_graph(k=10, max_distance_in_units=50)
        # g.plot(node_colors='white', edge_colors='white')
        # if 'graph' in s['imc']['transformed_mean']:
        #     del s['imc']['transformed_mean']['graph']
        # s['imc']['transformed_mean'].graph = g
##
if n_ or t_ or p_ and False:
    _, ax = plt.subplots(figsize=(10, 10))
    s['imc']['ome'].plot(0, ax=ax, preprocessing=np.arcsinh, cmap=matplotlib.cm.get_cmap('gray'))
    s['imc']['transformed_mean'].plot(0, ax=ax)
    s['imc']['transformed_mean'].graph.plot(node_colors='w', node_size=1, edge_colors=[1., 1., 1., 0.5],
                                            edge_size=0.5, ax=ax)
    plt.show()
##
if n_ or t_ or p_ and False:
    _, ax = plt.subplots(figsize=(10, 10))
    s['imc']['ome'].plot(0, ax=ax, preprocessing=np.arcsinh, cmap=matplotlib.cm.get_cmap('gray'))
    s['imc']['transformed_mean'].plot(0, fill_color=None, outline_color='channel', ax=ax)
    s['imc']['transformed_mean'].graph.plot(node_colors='w', node_size=1, edge_colors=[1., 1., 1., 0.5],
                                            edge_size=0.5, ax=ax)
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
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

        names_length_map = {}
        for i in range(len(self.filenames)):
            s = get_smu_file(self.split, i)
            n = len(s['imc']['transformed_mean'])
            names_length_map[i] = n
        self.map_left, self.map_right = get_bimap(names_length_map)

    @property
    def processed_file_names(self):
        return ["subgraphs.pt"]

    # def _compute_and_save(self, cell_index: int):
    #     subgraph = self.compute_subgraph(cell_index=cell_index)
    #     subgraph.dump()
    #     self.pbar.update(1)

    def len(self):
        return len(self.map_left)

    def process(self):
        data_list = []
        for i in range(len(self.filenames)):
            s = get_smu_file(self.split, i)
            g = s['imc']['transformed_mean'].graph
            n = len(s['imc']['transformed_mean'])
            for j in range(n):
                sub_g = g.subgraph_of_neighbors(node_index=j, subset_method='knn')
                print('oooooooo')
                print('oooooooo')
                print('oooooooo')
                print('oooooooo')
                # sub_data = Data(
                #     edge_index=sub_g.,
                #     edge_attr=sub_edge_attr,
                #     regions_centers=data.regions_centers[is_near],
                #     center_index=center_index,
                #     is_near=is_near,
                # )
                sub_data.num_nodes = len(sub_data.regions_centers)
                print('ooo')
                print('ooo')
                print('ooo')
                print('ooo')



        # for cell_index in tqdm(range(self.cells_count)):
        #     data = self.compute_subgraph(cell_index)
        #     data_list.append(data)
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
            g = to_networkx(
                data, edge_attrs=["edge_attr"], node_attrs=["regions_centers"]
            )
            neighbors = list(g.neighbors(local_cell_index))
            l = []
            for node in neighbors:
                w = g.get_edge_data(local_cell_index, node)["edge_attr"]
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

cell_graphs_dataset = CellGraphsDataset(split='train', name='knn_10_max_distance_in_units_50')
##
