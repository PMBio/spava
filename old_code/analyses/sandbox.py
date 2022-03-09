##
import pickle
from old_code.data2 import AccumulatedDataset, file_path, ExpressionFilteredDataset, FilteredMasksRelabeled
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import os
import torch
from torch import nn

f = file_path('accumulated_features/raw_accumulated.hdf5')
import h5py

with h5py.File(f, 'r') as f5:
    print(f5[f5.__iter__().__next__()]['region_center'])


class GraphDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.ds = AccumulatedDataset(split, feature='region_center', from_raw=True, transform=False)
        self.filenames = eval(self.split)
        assert len(self.ds) == len(self.filenames)
        self.index_converter = FilteredMasksRelabeled(split).get_indices_conversion_arrays
        self.knn_file = file_path(f'knn_indices_{split}.h5py')
        self.torch_graphs_file = file_path(f'torch_graphs_{split}.pickle')

        # computing/loading the knn indices
        # os.unlink(self.knn_file)
        if not os.path.isfile(self.knn_file):
            with h5py.File(self.knn_file, 'w') as f5:
                for i in tqdm(range(len(self.ds)), desc='computing knns'):
                    centers = self.ds[i]
                    new_centers = ExpressionFilteredDataset.expression_old_to_new(centers, i,
                                                                                  index_converter=self.index_converter)
                    # knn method
                    k = 20
                    neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(new_centers)
                    distances, indices = neighbors.kneighbors(new_centers)
                    # check that every element is a knn of itself, and this is the first neighbor
                    assert np.all(indices[:, 0] == np.array(range(len(new_centers))))
                    f = self.filenames[i]
                    f5[f'{f}/distances'] = distances
                    f5[f'{f}/indices'] = indices
                    f5[f'{f}/centers'] = new_centers

        # computing/loading the torch graphs
        # os.unlink(self.torch_graphs_file)
        if not os.path.isfile(self.torch_graphs_file):
            self.torch_graphs = []
            with h5py.File(self.knn_file, 'r') as f5:
                for i in tqdm(range(len(self.ds)), desc='building graphs'):
                    f = self.filenames[i]
                    distances = f5[f'{f}/distances'][...]
                    indices = f5[f'{f}/indices'][...]
                    centers = f5[f'{f}/centers'][...]
                    edges = []
                    weights = []
                    for k in range(indices.shape[0]):
                        for j in range(1, indices.shape[1]):
                            edge = [k, indices[k, j]]
                            weight = distances[k, j]
                            # weight = 1 / (distances[k, j] ** 2 + 1)
                            edges.append(edge)
                            weights.append(weight)
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(weights, dtype=torch.float).reshape((-1, 1))
                    threshold = 50
                    to_keep = (edge_attr < threshold).flatten()
                    # shape [n, 1]
                    edge_attr = edge_attr[to_keep, :]
                    # shape [2, n]
                    edge_index = edge_index[:, to_keep]
                    data = Data(edge_index=edge_index, edge_attr=edge_attr, centers=centers, num_nodes=len(centers))
                    self.torch_graphs.append(data)
            pickle.dump(self.torch_graphs, open(self.torch_graphs_file, 'wb'))
        else:
            self.torch_graphs = pickle.load(open(self.torch_graphs_file, 'rb'))

    def __len__(self):
        return len(self.torch_graphs)

    def __getitem__(self, i):
        return self.torch_graphs[i]


class GeometricPlainGnn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gcn0 = GCNConv(self.in_channels, self.in_channels)
        self.gcn1 = GCNConv(self.in_channels, self.in_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.gcn0(x, edge_index)
        x = self.relu(x)
        x = self.gcn1(x, edge_index)
        x = self.relu(x)
        return x


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(GCNConv, self).__init__(aggr=aggr)  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


ds = GraphDataset('train')
loader = DataLoader(ds, batch_size=16, num_workers=0, pin_memory=True, shuffle=True)
##
data = loader.__iter__().__next__()
##
model = GeometricPlainGnn(in_channels=39)
model.forward(x=, edge_index=data)