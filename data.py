a = '/data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/phyper_data/accumulated_features/b3f06e1c82889221ec4ac7c901afe5295666b7ab905716bf32072ba1e2920abb/cell_features.hdf5'

import h5py
import os
from torch.utils.data import Dataset
from splits import *
import vigra
import skimage
import skimage.io
import numpy as np
import torch

print(os.path.isfile(a))
with h5py.File(a, 'r') as f:
    print(f['BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_100_239_X12Y3_177_a0_full.tiff'].keys())


class OmeDataset(Dataset):
    # ome_normalization_method -1 for raw data
    def __init__(self, split):
        super().__init__()
        self.split = split
        assert split in ['train', 'validation', 'test']
        if split == 'train':
            self.filenames = train
        elif split == 'validation':
            self.filenames = validation
        elif split == 'test':
            self.filenames = test
        self.channels_count = self.__getitem__(0).shape[2]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        filename = self.filenames[i]
        f = os.path.join('data/OMEandSingleCellMasks/ome/', filename)
        ome = skimage.io.imread(f)
        ome = np.moveaxis(ome, 0, 2)
        ome = torch.from_numpy(ome).float()
        to_keep = [8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                   23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
        ome = ome[:, :, to_keep]
        ome = ome.float()
        return ome


CHANNEL_NAMES = ['H3tot', 'H3met', 'CK5', 'Fibronectin', 'CK19', 'CK8/18', 'TWIST1', 'CD68', 'CK14', 'SMA',
                 'Vimentin', 'Myc', 'HER2', 'CD3', 'H3phospho', 'ERK1/2', 'SLUG', 'ER', 'PR', 'p53', 'CD44',
                 'EpCAM', 'CD45', 'GATA3', 'CD20', 'betaCatenin', 'CAIX', 'Ecadherin', 'Ki67', 'EGFR', 'S6',
                 'Sox9', 'vWf_CD31', 'mTOR', 'CK7', 'panCK', 'cPARP_cCasp3', 'DNA1', 'DNA2']


class MasksDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        assert split in ['train', 'validation', 'test']
        if split == 'train':
            self.filenames = train
        elif split == 'validation':
            self.filenames = validation
        elif split == 'test':
            self.filenames = test

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        ome_filename = self.filenames[i]
        masks_file = '/data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/relabelled_masks.hdf5'
        with h5py.File(masks_file, 'r') as f5:
            masks = f5[ome_filename + '/masks'][...]
        return masks


class RawMeanDataset(Dataset):
    def __init__(self, split):
        self.split = split
        assert split in ['train', 'validation', 'test']
        if split == 'train':
            self.filenames = train
        elif split == 'validation':
            self.filenames = validation
        elif split == 'test':
            self.filenames = test
        # k is also a parameter, but only when having a gnn model, probably I should structure these check in a more
        # uniform way

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        with h5py.File(
                '/data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/phyper_data/accumulated_features/b3f06e1c82889221ec4ac7c901afe5295666b7ab905716bf32072ba1e2920abb/cell_features.hdf5',
                'r') as f5:
            x = f5[self.filenames[i] + f'/mean'][...]
            x = x.astype(np.float32)
            x = torch.from_numpy(x)
            return x
        # return self.scale(x, i)


class GraphDataset(Dataset):
    def __init__(self, split):
        self.split = split
        assert split in ['train', 'validation', 'test']
        if split == 'train':
            self.filenames = train
        elif split == 'validation':
            self.filenames = validation
        elif split == 'test':
            self.filenames = test

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        with h5py.File('/data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/region_centers.hdf5', 'r') as f5:
            # regions_centers = f5[self.filenames[i] + f'/region_center'][...]
            f = f'graphs/{self.filenames[i]}'
            data = torch.load(f)
            edge_index = data.edge_index
            return edge_index


if __name__ == '__main__':
    for f in [OmeDataset, MasksDataset, MeanDataset, GraphDataset]:
        for a in ['train', 'validation', 'test']:
            ds = f(a)
            print(ds[12].shape)
