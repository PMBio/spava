a = '/data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/phyper_data/accumulated_features/b3f06e1c82889221ec4ac7c901afe5295666b7ab905716bf32072ba1e2920abb/cell_features.hdf5'

import pickle
import h5py
import os
from torch.utils.data import Dataset
from splits import *
import skimage
import skimage.io
import numpy as np
import torch
import math
import cv2

import pathlib

current_file_path = pathlib.Path(__file__).parent.absolute()

channel_names = ['H3tot', 'H3met', 'CK5', 'Fibronectin', 'CK19', 'CK8/18', 'TWIST1', 'CD68', 'CK14', 'SMA',
                 'Vimentin', 'Myc', 'HER2', 'CD3', 'H3phospho', 'ERK1/2', 'SLUG', 'ER', 'PR', 'p53', 'CD44',
                 'EpCAM', 'CD45', 'GATA3', 'CD20', 'betaCatenin', 'CAIX', 'Ecadherin', 'Ki67', 'EGFR', 'S6',
                 'Sox9', 'vWf_CD31', 'mTOR', 'CK7', 'panCK', 'cPARP_cCasp3', 'DNA1', 'DNA2']


def file_path_old_data(f):
    return os.path.join(current_file_path, 'data/spatial_uzh_processed', f)


def file_path(f):
    return os.path.join(current_file_path, 'data/spatial_uzh_processed/a', f)


print(os.path.isfile(a))
with h5py.File(a, 'r') as f:
    print(f['BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_100_239_X12Y3_177_a0_full.tiff'].keys())

f = file_path(f'ok_cells_train.npy')
filter_expression_tensor_d_train = pickle.load(open(f, 'rb'))
f = file_path(f'ok_cells_validation.npy')
filter_expression_tensor_d_validation = pickle.load(open(f, 'rb'))
f = file_path(f'ok_cells_test.npy')
filter_expression_tensor_d_test = pickle.load(open(f, 'rb'))


def filter_expression_tensor(ome_filename, t, split):
    assert split in ['train', 'validation', 'test']
    if split == 'train':
        filter_expression_tensor_d = filter_expression_tensor_d_train
    if split == 'validation':
        filter_expression_tensor_d = filter_expression_tensor_d_validation
    if split == 'test':
        filter_expression_tensor_d = filter_expression_tensor_d_test
    list_of_cells = filter_expression_tensor_d['list_of_cells']
    list_of_ome_filenames = filter_expression_tensor_d['list_of_ome_filenames']
    list_of_ome_indices = filter_expression_tensor_d['list_of_ome_indices']
    list_of_cell_ids = filter_expression_tensor_d['list_of_cell_ids']
    cell_is_ok = filter_expression_tensor_d['cell_is_ok']
    begin = list_of_ome_filenames.index(ome_filename)
    end = len(list_of_ome_filenames) - list_of_ome_filenames[::-1].index(ome_filename)
    # print(list_of_ome_filenames[begin])
    # print(list_of_ome_filenames[end - 1])
    # print(list_of_ome_filenames[end])
    # print(list_of_ome_filenames[end + 1])
    oks = cell_is_ok[begin: end]
    # labels = list_of_cell_ids[begin: end]
    # print(oks.shape)
    # print(t.shape)
    ok = t[oks, :]
    return ok


def get_filtered_labels_mapping(ome_filename, split):
    f = file_path(f'ok_cells_{split}.npy')
    d = pickle.load(open(f, 'rb'))
    list_of_cells = d['list_of_cells']
    list_of_ome_filenames = d['list_of_ome_filenames']
    list_of_ome_indices = d['list_of_ome_indices']
    list_of_cell_ids = d['list_of_cell_ids']
    cell_is_ok = d['cell_is_ok']
    begin = list_of_ome_filenames.index(ome_filename)
    end = len(list_of_ome_filenames) - list_of_ome_filenames[::-1].index(ome_filename)
    # print(list_of_ome_filenames[begin])
    # print(list_of_ome_filenames[end - 1])
    # print(list_of_ome_filenames[end])
    # print(list_of_ome_filenames[end + 1])
    oks = cell_is_ok[begin: end]
    # labels = list_of_cell_ids[begin: end]
    # print(oks.shape)
    # print(t.shape)
    l0 = list(range(end - begin))
    l1 = list_of_cell_ids
    print(f'l0 = {l0}, l1 = {l1}')
    d = dict(zip(l0, l1))
    # ok = t[oks, :]
    return d


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


class OmeDatasetHotPixelsRemoved(Dataset):
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
        to_keep = [8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                   23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
        ome = ome[:, :, to_keep]

        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        maxs = cv2.dilate(ome, kernel, iterations=1, borderType=cv2.BORDER_REFLECT101)
        mask = ome - maxs >= 50
        ome[mask] = maxs[mask]

        ome = torch.from_numpy(ome).float()
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
        masks_file = file_path_old_data('relabelled_masks.hdf5')
        with h5py.File(masks_file, 'r') as f5:
            masks = f5[ome_filename + '/masks'][...]
        return masks


class RawDataset(Dataset):
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

    def get_item(self, i, feature):
        f = file_path_old_data('phyper_data/accumulated_features/b3f06e1c82889221ec4ac7c901afe5295666b7ab905716bf32072ba1e2920abb/cell_features.hdf5')
        with h5py.File(f, 'r') as f5:
            x = f5[self.filenames[i] + f'/{feature}'][...]
            x = x.astype(np.float32)
            x = torch.from_numpy(x)
            return x
        # return self.scale(x, i)


class RawCountDataset(RawDataset):
    def __init__(self, split):
        super().__init__(split)

    def __getitem__(self, item):
        t = self.get_item(item, 'count')
        ome_filename = self.filenames[item]
        filtered_t = filter_expression_tensor(ome_filename, t, self.split)
        return filtered_t


class AreaDataset(RawDataset):
    def __init__(self, split):
        super().__init__(split)
        self.masks_dataset = MasksDataset(split)

    def __getitem__(self, item):
        masks = self.masks_dataset[item]
        label, count = np.unique(masks.ravel(), return_counts=True)
        return count


class RawMeanDataset(RawDataset):
    def __init__(self, split):
        super().__init__(split)

    def __getitem__(self, item):
        t = self.get_item(item, 'mean')
        ome_filename = self.filenames[item]
        import time
        # start = time.time()
        filtered_t = filter_expression_tensor(ome_filename, t, self.split)
        # print(f'filtering: {time.time() - start}')
        return filtered_t


class TransformedMeanDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.raw_mean_dataset = RawMeanDataset(split)
        f = file_path(f'scaler_{split}.pickle')
        self.d = pickle.load(open(f, 'rb'))
        self.filenames = self.raw_mean_dataset.filenames

    def __len__(self):
        return len(self.raw_mean_dataset)

    def __getitem__(self, item):
        o = self.filenames[item]
        q0 = self.d[o]['q0']
        q1 = self.d[o]['q1']
        t = self.raw_mean_dataset[item]
        t0 = t / q0
        cofactor = q1 / math.sinh(1)
        # not tested yet
        z = torch.asinh(t0 / cofactor)
        # z = np.arcsinh(t0 / cofactor)
        assert type(z) == torch.Tensor
        if z.dtype != torch.float:
            z = z.float()
        return z
        # return torch.tensor(z, dtype=torch.float)


class RawMean12(RawDataset):
    def __init__(self, split):
        super().__init__(split)
        self.raw_mean_ds = RawMeanDataset(split)

    def __getitem__(self, item):
        t = self.raw_mean_ds[item]
        return torch.asinh(t)


class NatureBOriginal(RawDataset):
    def __init__(self, split):
        super().__init__(split)
        self.ds = RawMeanDataset(split)
        self.list_of_z = []
        self.list_of_patient_index = []
        from tqdm import tqdm
        for i, x in enumerate(tqdm(self.ds)):
            q0 = np.quantile(x.numpy(), q=0.99, axis=0)
            x0 = x / q0
            x0 = torch.tensor(x0, dtype=torch.float)
            ii = torch.tensor([i] * len(x))
            self.list_of_patient_index.append(ii)
            self.list_of_z.append(x0)
        self.t_train_raw2 = torch.cat(self.list_of_z, dim=0)
        self.patient_indexes = torch.cat(self.list_of_patient_index)

    def __getitem__(self, item):
        t = self.list_of_z[item]
        return t


class NatureBImproved(RawDataset):
    def __init__(self, split):
        super().__init__(split)
        self.nature_b_ds = NatureBOriginal(split)

    def __getitem__(self, item):
        t = self.nature_b_ds[item]
        return torch.asinh(t)


# class GraphDataset(Dataset):
#     def __init__(self, split):
#         self.split = split
#         assert split in ['train', 'validation', 'test']
#         if split == 'train':
#             self.filenames = train
#         elif split == 'validation':
#             self.filenames = validation
#         elif split == 'test':
#             self.filenames = test
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, i):
#         with h5py.File('/data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/region_centers.hdf5', 'r') as f5:
#             # regions_centers = f5[self.filenames[i] + f'/region_center'][...]
#             f = f'graphs/{self.filenames[i]}'
#             data = torch.load(f)
#             edge_index = data.edge_index
#             return edge_index


if __name__ == '__main__':
    # ds = TransformedMeanDataset('train')
    ds = OmeDatasetHotPixelsRemoved('train')
    print(ds[2].shape)
    # ds = RawMeanDataset('train')
    # print(ds[2].shape)
    # for f in [OmeDataset, MasksDataset, RawMeanDataset, GraphDataset, AreaDataset]:
    #     for a in ['train', 'validation', 'test']:
    #         ds = f(a)
    #         print(ds[12].shape)
