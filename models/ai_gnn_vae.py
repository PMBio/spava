import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import DataLoader as GeometricDataLoader
from graphs import CellExpressionGraph
from torch_geometric.nn.conv import GINEConv, GINEConv


class GeometricPlainGnn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        gine0_nn = nn.Sequential(nn.Linear(self.in_channels, self.in_channels),
                                      nn.ReLU(),
                                      nn.Linear(self.in_channels, self.in_channels),
                                      nn.ReLU())
        gine1_nn = nn.Sequential(nn.Linear(self.in_channels, self.in_channels),
                                      nn.ReLU(),
                                      nn.Linear(self.in_channels, self.in_channels),
                                      nn.ReLU())
        self.gcn0 = GINEConv(gine0_nn)
        self.gcn1 = GINEConv(gine1_nn)
        self.linear0 = nn.Linear(1, self.in_channels)
        self.linear1 = nn.Linear(self.in_channels, self.in_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        e = self.linear1(self.relu(self.linear0(edge_attr)))
        x = self.gcn0(x, edge_index, e)
        x = self.relu(x)
        x = self.gcn1(x, edge_index, e)
        x = self.relu(x)
        return x


##
ds = CellExpressionGraph("validation", "gaussian")
loader = GeometricDataLoader(ds, batch_size=32, shuffle=True)
##
model = GeometricPlainGnn(in_channels=39)
data = loader.__iter__().__next__()
output = model(data.x, data.edge_index, data.edge_attr)