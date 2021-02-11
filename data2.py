import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import skimage
import skimage.io
import torch
import cv2
import vigra
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import splits
import matplotlib
import os
import torch
from torch.utils.data import DataLoader  # SequentialSampler

import pathlib

current_file_path = pathlib.Path(__file__).parent.absolute()
# current_file_path = os.path.abspath('')


def file_path_old_data(f):
    return os.path.join(current_file_path, 'data/spatial_uzh_processed', f)


def file_path(f):
    return os.path.join(current_file_path, 'data/spatial_uzh_processed/a', f)


CHANNEL_NAMES = ['H3tot', 'H3met', 'CK5', 'Fibronectin', 'CK19', 'CK8/18', 'TWIST1', 'CD68', 'CK14', 'SMA',
                 'Vimentin', 'Myc', 'HER2', 'CD3', 'H3phospho', 'ERK1/2', 'SLUG', 'ER', 'PR', 'p53', 'CD44',
                 'EpCAM', 'CD45', 'GATA3', 'CD20', 'betaCatenin', 'CAIX', 'Ecadherin', 'Ki67', 'EGFR', 'S6',
                 'Sox9', 'vWf_CD31', 'mTOR', 'CK7', 'panCK', 'cPARP_cCasp3', 'DNA1', 'DNA2']

CHANNELS_TO_KEEP = [8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]

assert len(CHANNEL_NAMES) == len(CHANNELS_TO_KEEP)


def get_split(split):
    assert split in ['train', 'validation', 'test']
    if split == 'train':
        return splits.train
    elif split == 'validation':
        return splits.validation
    else:
        return splits.test


class MasksDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        self.filenames = get_split(self.split)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        ome_filename = self.filenames[i]
        masks_file = file_path_old_data('relabelled_masks.hdf5')
        with h5py.File(masks_file, 'r') as f5:
            masks = f5[ome_filename + '/masks'][...]
        return masks


class OmeDataset(Dataset):
    def __init__(self, split, hot_pixel_filtering=True):
        super().__init__()
        self.split = split
        self.filenames = get_split(self.split)
        self.channels_count = len(CHANNELS_TO_KEEP)
        self.hot_pixel_filtering = hot_pixel_filtering

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        filename = self.filenames[i]
        f = os.path.join('data/OMEandSingleCellMasks/ome/', filename)
        ome = skimage.io.imread(f)
        ome = np.moveaxis(ome, 0, 2)
        ome = ome[:, :, CHANNELS_TO_KEEP]

        if self.hot_pixel_filtering:
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            maxs = cv2.dilate(ome, kernel, iterations=1, borderType=cv2.BORDER_REFLECT101)
            mask = ome - maxs >= 50
            c = ome[mask] - maxs[mask]
            a = np.sum(c)
            b = np.sum(ome)
            ome[mask] = maxs[mask]

        ome = torch.from_numpy(ome).float()
        return ome


p = file_path('accumulated_features')
os.makedirs(p, exist_ok=True)


def save_features(features, f5):
    f5['count'] = features['Count'][1:]
    f5['maximum'] = features['Maximum'][1:]
    f5['mean'] = features['Mean'][1:]
    f5['sum'] = features['Sum'][1:]
    f5['variance'] = features['Variance'][1:]
    f5['region_center'] = features['RegionCenter'][1:]
    if 'Median' in features:
        f5['median'] = features['Median'][1:]


def extract_features_for_ome(ome, masks, compute_median=False):
    ome = np.require(ome, requirements=['C'])
    vigra_ome = vigra.taggedView(ome, 'xyc')
    masks = masks.astype(np.uint32)
    features = ['Count', 'Maximum', 'Mean', 'Sum', 'Variance', 'RegionCenter']
    features = vigra.analysis.extractRegionFeatures(vigra_ome, labels=masks, ignoreLabel=0, features=features)
    features = {k: v for k, v in features.items()}
    if compute_median:
        labels = np.unique(masks)
        medians = []
        for label in tqdm(labels, desc='computing median', leave=False):
            medians_cell = []
            for i in range(ome.shape[2]):
                m = np.median(ome[:, :, i][masks == label])
                medians_cell.append(m)
            medians.append(medians_cell)
        features['Median'] = np.array(medians)
        assert features['Median'].shape == features['Mean'].shape
    return features


class AccumulatedDataset(Dataset):
    def __init__(self, split, feature: str, from_raw: bool, transform: bool):
        self.split = split
        self.filenames = get_split(self.split)
        if from_raw:
            self.f = file_path('accumulated_features/raw_accumulated.hdf5')
        else:
            self.f = file_path('accumulated_features/transformed_accumulated.hdf5')
        self.feature = feature
        self.transform = transform
        assert not (not from_raw and transform)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        with h5py.File(self.f, 'r') as f5:
            x = f5[self.filenames[i] + f'/{self.feature}'][...]
            x = x.astype(np.float32)
            if self.transform:
                x = np.arcsinh(x)
            x = torch.from_numpy(x)
            return x


class ScaledAccumulatedDataset(Dataset):
    def __init__(self, split, feature: str, from_raw: bool, transform: bool):
        self.accumulated_ds = AccumulatedDataset(split, feature, from_raw, transform)
        self.mu, self.std = self.get_mean_and_std(self.accumulated_ds)

    def __len__(self):
        return len(self.accumulated_ds)

    @staticmethod
    def get_mean_and_std(ds):
        l = []
        for x in ds:
            l.append(x.numpy())
        z = np.concatenate(l, axis=0)
        mu = np.mean(z, axis=0)
        std = np.std(z, axis=0)
        return mu, std

    def scale(self, x):
        return (x - self.mu) / self.std

    def scale_back(self, z):
        return self.mu + self.std * z

    def __getitem__(self, i):
        x = self.accumulated_ds[i]
        z = self.scale(x)
        return z


class IndexInfo:
    def __init__(self, split):
        self.split = split
        self.filenames = get_split(self.split)

        self.unfiltered_begins = []
        self.unfiltered_ends = []
        self.filtered_begins = []
        self.filtered_ends = []
        ds = MasksDataset(self.split)
        self.ok_size_cells = pickle.load(open(file_path(f'ok_size_cells_{self.split}.npy'), 'rb'))
        i = 0
        j = 0
        for ome_index, masks in enumerate(ds):
            m = masks.max()
            begin = i
            end = begin + m
            i = end
            self.unfiltered_begins.append(begin)
            self.unfiltered_ends.append(end)

            ok = self.ok_size_cells[begin: end]
            begin = j
            end = begin + np.sum(ok)
            j = end
            self.filtered_begins.append(begin)
            self.filtered_ends.append(end)


class FilteredMasksDataset(Dataset):
    def __init__(self, split):
        self.masks_ds = MasksDataset(split)
        self.index_info = IndexInfo(split)

    def __len__(self):
        return len(self.masks_ds)

    def __getitem__(self, i):
        masks = self.masks_ds[i]
        m = masks.max()
        v = np.arange(1, m + 1, dtype=np.int)
        begin = self.index_info.unfiltered_begins[i]
        end = self.index_info.unfiltered_ends[i]
        ok = self.index_info.ok_size_cells[begin: end]
        ok_labels = v[ok]
        not_ok = set(v.tolist()).difference(ok_labels.tolist())
        for label in not_ok:
            masks[masks == label] = 0
        return masks


class FilteredMasksRelabeled(Dataset):
    def __init__(self, split):
        self.masks_ds = MasksDataset(split)
        self.index_info = IndexInfo(split)
        self.filtered_masks = FilteredMasksDataset(split)

    def __len__(self):
        return len(self.masks_ds)

    def get_indices_conversion_arrays(self, i):
        u = np.unique(self.filtered_masks[i])
        new_to_old = np.sort(u)
        m = np.max(self.masks_ds[i])
        old_to_new = np.zeros(m + 1, dtype=np.int)
        for uu in u:
            i, = np.where(new_to_old == uu)
            old_to_new[uu] = i
        return new_to_old, old_to_new

    def __getitem__(self, i):
        new_to_old, old_to_new = self.get_indices_conversion_arrays(i)
        old = self.filtered_masks[i]
        new = old_to_new[old]
        return new


class ExpressionDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.ds = ScaledAccumulatedDataset(split, feature='mean', from_raw=True, transform=True)
        self.index_converter = FilteredMasksRelabeled(split).get_indices_conversion_arrays

    # the background is not an expression rows, so for instance there is no new row which corresponds to the background in the old cell labeling. The asserts will check that everything is ok
    def expression_old_to_new(self, old_e, i):
        new_to_old, old_to_new = self.index_converter(i)
        assert len(old_e) == len(old_to_new) - 1, (len(old_e), len(old_to_new) + 1, f'ome_index{i}')
        new = []
        for i in range(len(old_to_new)):
            o = old_to_new[i]
            if o == 0:
                continue
            new.append(old_e[(i - 1,), :])
        new_e = np.concatenate(new, axis=0)
        assert len(new_e) == len(new_to_old) - 1, f'len(new_e) = {len(new_e)}, len(new_to_old) = {len(new_to_old)}'
        return new_e

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        e = self.ds[i]
        new_e = self.expression_old_to_new(e, i)
        return new_e


class CenterFilteredDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.ds = ExpressionDataset(split)
        self.filenames = get_split(self.split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        f_in = file_path('accumulated_features/raw_accumulated.hdf5')
        with h5py.File(f_in, 'r') as f5:
            o = self.filenames[i]
            e = f5[f'{o}/region_center'][...]
        new_e = self.ds.expression_old_to_new(e, i)
        return new_e


class CellDataset(Dataset):
    def __init__(self, split, features=None):
        self.features = features or {'expression': True, 'center': False, 'ome': True, 'mask': True}
        self.split = split
        with h5py.File(file_path('merged_filtered_centers_and_expressions.hdf5'), 'r') as f5:
            self.expressions = f5[f'{split}/expressions'][...]
            self.centers = f5[f'{split}/centers'][...]
        self.f5 = h5py.File(file_path('filtered_cells_dataset.hdf5'), 'r')
        self.f5_omes = self.f5[f'{split}/omes']
        self.f5_masks = self.f5[f'{split}/masks']
        assert len(self.expressions) == len(self.f5_omes)
        assert len(self.expressions) == len(self.f5_masks)

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, i):
        l = []
        if self.features['expression']:
            l.append(self.expressions[i])
        if self.features['center']:
            l.append(self.centers[i])
        if self.features['ome']:
            l.append(self.f5_omes[f'{i}'][...])
        if self.features['mask']:
            l.append(self.f5_masks[f'{i}'][...])
        return l


assert torch.cuda.is_available()

if __name__ == '__main__':
    dataset = CellDataset('train')
    loader = DataLoader(dataset, batch_size=1024, num_workers=16, pin_memory=True, shuffle=True)
    with h5py.File(file_path('filtered_cells_dataset.hdf5'), 'r') as f5:
        for cell_k in [0, 1, 2, 23, 123, 12, 412, 4]:
            x = f5[f'train/omes/{cell_k}'][...]
            s = x.shape
            x = x.reshape((-1, 39))
            pca = PCA(3).fit_transform(x)
            a = pca.min(axis=0)
            b = pca.max(axis=0)
            pca = (pca - a) / (b - a)
            print(pca.shape)
            print(pca.min(), pca.max())
            pca.shape = [s[0], s[1], 3]
            plt.figure()
            plt.subplot(1, 2, 1)
            #    plt.imshow(pca)
            x.shape = s
            plt.imshow(x[:, :, 34])
            plt.subplot(1, 2, 2)
            plt.imshow(f5[f'train/masks/{cell_k}'][...])
            plt.show()

    with h5py.File(file_path('filtered_cells_dataset.hdf5'), 'r') as f5:
        cell_k = 10000
        x = f5[f'train/omes/{cell_k}'][...]
        plt.imshow(f5[f'train/masks/{cell_k}'][...])
        axes = plt.subplots(8, 5, figsize=(5 * 2, 8 * 1.8))[1].flatten()
        for i in range(39):
            axes[i].imshow(x[:, :, i], cmap=matplotlib.cm.get_cmap('gray'))
        plt.show()
