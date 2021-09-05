##
import random

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import skimage
import skimage.io
import torch
import cv2
import torchvision.transforms
import vigra
# from tqdm.notebook import tqdm
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import splits
import matplotlib
import os
import torch
from torch.utils.data import DataLoader  # SequentialSampler

import pathlib

try:
    current_file_path = pathlib.Path(__file__).parent.absolute()


    def file_path(f):
        return os.path.join(current_file_path, 'data/spatial_uzh_processed/a', f)
except NameError:
    print('setting data path manually')


    def file_path(f):
        return os.path.join('/data/l989o/data/basel_zurich/spatial_uzh_processed/a', f)

if __name__ == '__main__':
    # PLOT = True
    PLOT = False
    COMPUTE = False
    # DEBUG = True
    DEBUG = False
else:
    PLOT = False
    COMPUTE = False
    DEBUG = False

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


##

class MasksDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        self.filenames = get_split(self.split)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        ome_filename = self.filenames[i]
        masks_file = file_path('../relabelled_masks.hdf5')
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
        f = file_path(os.path.join('../../OMEandSingleCellMasks/ome/', filename))
        ome = skimage.io.imread(f)
        ome = np.moveaxis(ome, 0, 2)
        ome = ome[:, :, CHANNELS_TO_KEEP]

        if self.hot_pixel_filtering:
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            maxs = cv2.dilate(ome, kernel, iterations=1, borderType=cv2.BORDER_REFLECT101)
            mask = ome - maxs >= 50
            # c = ome[mask] - maxs[mask]
            # a = np.sum(c)
            # b = np.sum(ome)
            ome[mask] = maxs[mask]

        ome = torch.from_numpy(ome).float()
        return ome


##

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
            raise RuntimeError('case not longer supported')
            # self.f = file_path('accumulated_features/transformed_accumulated.hdf5')
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


if COMPUTE:
    p = file_path('accumulated_features')
    os.makedirs(p, exist_ok=True)

    f = os.path.join(p, 'raw_accumulated.hdf5')

    with h5py.File(f, 'w') as f5:
        def accumulate_split(split):
            ome_dataset = OmeDataset(split)
            masks_dataset = MasksDataset(split)
            assert len(ome_dataset) == len(masks_dataset)
            for ome_filename, ome, masks in tqdm(zip(ome_dataset.filenames, ome_dataset, masks_dataset),
                                                 total=len(ome_dataset),
                                                 desc=f'accumulating {split} set'):
                g5 = f5.create_group(ome_filename)
                features = extract_features_for_ome(ome, masks)
                save_features(features, g5)


        accumulate_split('train')
        accumulate_split('validation')
        accumulate_split('test')


##
def plot_expression_dataset(ds):
    l = []
    for e in ds:
        l.append(e.mean(dim=0, keepdim=True))
    ee = torch.cat(l, dim=0)
    print(ee.shape)
    plt.figure()
    plt.imshow(ee.numpy())
    plt.show()


if PLOT:
    ds = AccumulatedDataset('validation', 'mean', from_raw=True, transform=True)
    plot_expression_dataset(ds)


##
# this function is called more times than needed, but performance is still fine
def get_ok_size_cells(split):
    areas_ds = AccumulatedDataset(split, 'count', from_raw=True, transform=False)
    all_areas = []
    # note that we remove the background before concatenating
    check = 0
    # for area in tqdm(areas_ds, desc=f'filtering cells by areas, {split} set'):
    for area in areas_ds:
        all_areas.append(area)
        check += len(area)

    areas = torch.cat(all_areas)
    assert len(areas) == check
    print(len(areas), 'cells in', split, 'set')

    min_area = 15.
    max_area = 400.
    if PLOT:
        plt.figure()
        plt.hist(areas.numpy(), bins=100)
        plt.axvline(x=min_area, c='r')
        plt.axvline(x=max_area, c='r')
        plt.xlabel('cell area')
        plt.ylabel('count')
        plt.title(f'distribution of cell area, {split} set')
        plt.show()

    a = areas.numpy()
    ok_size_cells = np.logical_and(a >= min_area, a <= max_area)
    return ok_size_cells


if PLOT:
    _ = get_ok_size_cells('train')
    _ = get_ok_size_cells('validation')
    _ = get_ok_size_cells('test')


##
# this class refers to accumulated quantities, which do not include the background, so it is like merging together
# all the cells that are not background
class IndexInfo:
    def __init__(self, split):
        self.split = split
        self.filenames = get_split(self.split)

        self.unfiltered_begins = []
        self.unfiltered_ends = []
        self.filtered_begins = []
        self.filtered_ends = []
        ds = MasksDataset(self.split)
        self.ok_size_cells = get_ok_size_cells(split)
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


# this approach is horrible, so I'll try to find the time to replace it with "spatial muon"
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


if PLOT:
    ds0 = MasksDataset('train')
    ds1 = FilteredMasksRelabeled('train')
    ome_index = 42
    x0 = ds0[ome_index]
    x1 = ds1[ome_index]
    x0[x0 > 0] = 1
    differences = (x1 == 0) & (x1 != x0)
    print(f'fraction of filtered pixels {np.sum(differences) / np.prod(differences.shape):0.4f}')
    plt.figure(figsize=(10, 5))
    plt.imshow(x0)
    redify = np.array([[0., 0., 0., 0.], [1., 0., 0., 1.]])
    plt.imshow(redify[differences.astype(np.int)])
    plt.show()


##
def prepend_background(t, background_value=0):
    assert len(t.shape) == 1
    if type(t) == np.ndarray:
        new_t = np.concatenate((np.zeros_like(t)[:1], t))
        new_t[0] = background_value
    else:
        assert type(t) == torch.Tensor
        raise RuntimeError('WeirdError: this code was working yesterday and today started giving me a strange '
                           'exception. I\'ll leave it sit for a while before debugging it, for now just numpy stuff')
        new_t = torch.cat((torch.zeros_like(t)[:1], t))
        new_t[0] = background_value
    return new_t


if PLOT:
    print(prepend_background(np.array([3, 4, 5])))
    # print(prepend_background(torch.tensor([3.2, 3.4]), background_value=-2))
##

if PLOT:
    split = 'train'
    ds = AccumulatedDataset(split, 'mean', from_raw=True, transform=True)
    l = []
    for x in ds:
        l.append(x)
    expressions = torch.cat(l, dim=0).numpy()
    ok_cells = get_ok_size_cells(split)
    ome_index = 1
    m0 = MasksDataset(split)[ome_index]
    m1 = FilteredMasksRelabeled(split)[ome_index]
    e0 = ds[ome_index].numpy()
    ii = IndexInfo(split)
    begin = ii.filtered_begins[ome_index]
    end = ii.filtered_ends[ome_index]
    e1 = expressions[ok_cells][begin: end]
    channel = 4

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    v0 = np.min([np.min(e0[:, channel]), np.min(e1[: channel])])
    v1 = np.max([np.max(e0[:, channel]), np.max(e1[: channel])])
    im0 = prepend_background(e0[:, channel], background_value=v0)[m0]
    im1 = prepend_background(e1[:, channel], background_value=v0)[m1]
    sm0 = axes[0].imshow(im0)
    sm1 = axes[1].imshow(im1)
    sm0.set_clim([v0, v1])
    sm1.set_clim([v0, v1])

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sm0, cax=cax0, orientation='vertical')

    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(sm1, cax=cax1, orientation='vertical')

    filtered = ((m1 == 0) & (m1 != m0)).astype(np.int)
    redify = np.array([[0., 0., 0., 0.], [1., 0., 0., 1.]])
    axes[1].imshow(redify[filtered])

    plt.tight_layout()
    plt.subplots_adjust()
    plt.show()


##
class ExpressionFilteredDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.ds = AccumulatedDataset(split, feature='mean', from_raw=True, transform=True)
        self.index_converter = FilteredMasksRelabeled(split).get_indices_conversion_arrays

    # allows for easy filtering of quantities that are defined for each cell, so to match the indexing given by
    # FilteredMasksRelabeled
    #
    # the background is not an expression rows, so for instance there is no new row which corresponds to the
    # background in the old cell labeling. The asserts will check that everything is ok
    @staticmethod
    def expression_old_to_new(old_e, i, index_converter):
        new_to_old, old_to_new = index_converter(i)
        # both old_e and new_e don't have the background, but new_to_old and old_to_new takes into account for the
        # background, since when dealing with masks index, the zero is teh background. It's a mess I know,
        # I'll probably replace this with "spatial muon"
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
        new_e = ExpressionFilteredDataset.expression_old_to_new(e, i, self.index_converter)
        return new_e


class CenterFilteredDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.ds = ExpressionFilteredDataset(split)
        self.filenames = get_split(self.split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        f_in = file_path('accumulated_features/raw_accumulated.hdf5')
        with h5py.File(f_in, 'r') as f5:
            o = self.filenames[i]
            e = f5[f'{o}/region_center'][...]
        new_e = self.ds.expression_old_to_new(e, i, self.ds.index_converter)
        return new_e


##
COMPUTE = True
PLOT = True
DEBUG = True
from scipy.ndimage import center_of_mass

if COMPUTE:
    f_in = file_path('accumulated_features/raw_accumulated.hdf5')
    f_out = file_path('filtered_cells_dataset.hdf5')

    if DEBUG:
        lists_of_centers = {'train': [], 'validation': [], 'test': []}
        lists_of_expressions = {'train': [], 'validation': [], 'test': []}

    if DEBUG and PLOT:
        DEBUG_WITH_PLOTS = True
    else:
        DEBUG_WITH_PLOTS = False

    with h5py.File(f_out, 'w') as f5_out:
        with h5py.File(f_in, 'r') as f5_in:
            for split in tqdm(['train', 'validation', 'test'], desc=f'split'):
                k = 0
                ds = ExpressionFilteredDataset(split)
                ome_ds = OmeDataset(split)
                masks_ds = FilteredMasksRelabeled(split)
                old_masks_ds = MasksDataset(split)  # to debug
                for ome_index, e in enumerate(tqdm(ds, desc=f'isolating all cells {split}')):
                    ome = ome_ds[ome_index].numpy()
                    masks = masks_ds[ome_index]
                    old_masks = old_masks_ds[ome_index]  # to debug
                    new_to_old, old_to_new = masks_ds.get_indices_conversion_arrays(ome_index)
                    for cell_index in range(len(e)):
                        o = ome_ds.filenames[ome_index]
                        # + 1 because in the expression data the background has been removed
                        new = cell_index + 1
                        old = new_to_old[new].item()
                        # - 1 because in the hdf5 file the background has been removed
                        center = f5_in[f'/{o}/region_center'][old - 1, :]
                        # compute the bounding box of the mask
                        z = masks == new
                        p0 = np.sum(z, axis=0)
                        p1 = np.sum(z, axis=1)
                        w0, = np.where(p0 > 0)
                        w1, = np.where(p1 > 0)
                        a0 = w0[0]
                        b0 = w0[-1] + 1
                        a1 = w1[0]
                        b1 = w1[-1] + 1
                        y = z[a1: b1, a0: b0]
                        if DEBUG:
                            z_center = center_of_mass(z)
                        if DEBUG_WITH_PLOTS:
                            plt.figure(figsize=(20, 20))
                            plt.imshow(z)
                            plt.scatter(z_center[1], z_center[0], color='red', s=1)
                            plt.show()
                        if DEBUG:
                            center_debug = np.array(center_of_mass(old_masks == old))
                        if DEBUG_WITH_PLOTS:
                            plt.figure()
                            plt.imshow(old_masks == old)
                            plt.scatter(center[1], center[0], color='red', s=1)
                            plt.scatter(center_debug[1], center_debug[0], color='green', s=1)
                            plt.show()

                        y_center = np.array(center_of_mass(y))
                        if DEBUG_WITH_PLOTS:
                            plt.figure()
                            plt.imshow(y)
                            plt.scatter(y_center[1], y_center[0], color='black', s=1)
                            plt.show()
                        if DEBUG:
                            assert np.allclose(center, z_center)
                        r = 15
                        l = 2 * r + 1
                        square_ome = np.zeros((l, l, ome.shape[2]))
                        square_mask = np.zeros((l, l))


                        def get_coords_for_padding(des_r, src_shape, src_center):
                            des_l = 2 * des_r + 1

                            def f(src_l, src_c):
                                a = src_c - des_r
                                b = src_c + des_r
                                if a < 0:
                                    c = - a
                                    a = 0
                                else:
                                    c = 0
                                if b > src_l:
                                    b = src_l
                                src_a = a
                                src_b = b
                                des_a = c
                                des_b = des_a + b - a
                                return src_a, src_b, des_a, des_b

                            src0_a, src0_b, des0_a, des0_b = f(src_shape[0], int(src_center[0]))
                            src1_a, src1_b, des1_a, des1_b = f(src_shape[1], int(src_center[1]))
                            return src0_a, src0_b, src1_a, src1_b, des0_a, des0_b, des1_a, des1_b


                        src0_a, src0_b, src1_a, src1_b, des0_a, des0_b, des1_a, des1_b = get_coords_for_padding(r,
                                                                                                                y.shape,
                                                                                                                y_center)

                        square_ome[des0_a:des0_b, des1_a:des1_b, :] = ome[a1:b1, a0:b0, :][src0_a:src0_b, src1_a:src1_b,
                                                                      :]
                        square_mask[des0_a:des0_b, des1_a:des1_b] = y[src0_a:src0_b, src1_a:src1_b]

                        if DEBUG_WITH_PLOTS:
                            plt.figure()
                            plt.imshow(square_mask)
                            plt.scatter(r, r, color='blue', s=1)
                            plt.show()
                        #                     f5_out[f'{split}/omes/{k}'] = ome[a1: b1, a0: b0]
                        #                     f5_out[f'{split}/masks/{k}'] = y
                        f5_out[f'{split}/omes/{k}'] = square_ome
                        f5_out[f'{split}/masks/{k}'] = square_mask
                        if DEBUG:
                            lists_of_centers[split].append(center)
                            lists_of_expressions[split].append(e[cell_index])
                        k += 1
                        if DEBUG_WITH_PLOTS:
                            if k >= 4:
                                DEBUG_WITH_PLOTS = False
                                import sys
                                sys.exit(0)

# %%
#
# from sklearn.decomposition import PCA
#
# if DEBUG:
#     k = 2
#     with h5py.File(file_path('filtered_cells_dataset.hdf5'), 'r') as f5:
#         ds = ExpressionFilteredDataset('train')
#         l = []
#         for e in tqdm(ds):
#             l.append(e)
#         e = np.concatenate(l, axis=0)
#         len(e)
#         ee = e[:k, ]
#         llll = []
#         for lll in tqdm(lists_of_expressions['train']):
#             llll.append(lll.reshape((1, -1)))
#         eee = np.concatenate(llll, axis=0)
#         index_info = IndexInfo('train')
#         assert np.all(ee == eee)
#         print('expressions computed correctly')
#
#         ds = CenterFilteredDataset('train')
#         l = []
#         for e in tqdm(ds):
#             l.append(e)
#         e = np.concatenate(l, axis=0)
#         len(e)
#         ee = e[:k, ]
#         llll = []
#         for lll in tqdm(lists_of_centers['train']):
#             llll.append(lll.reshape((1, -1)))
#         eee = np.concatenate(llll, axis=0)
#         index_info = IndexInfo('train')
#         assert np.all(ee == eee)
#         print('region centers computed correctly')


##

import matplotlib

with h5py.File(file_path('filtered_cells_dataset.hdf5'), 'r') as f5:
    cell_k = 10000
    x = f5[f'train/omes/{cell_k}'][...]
    plt.imshow(f5[f'train/masks/{cell_k}'][...])
    plt.show()
    axes = plt.subplots(8, 5, figsize=(5 * 2, 8 * 1.8))[1].flatten()
    for i in range(39):
        axes[i].imshow(x[:, :, i], cmap=matplotlib.cm.get_cmap('gray'))
    plt.show()

##
if COMPUTE:
    with h5py.File(file_path('merged_filtered_centers_and_expressions.hdf5'), 'w') as f5:
        for split in ['train', 'validation', 'test']:
            filenames = get_split(split)
            expression_ds = ExpressionFilteredDataset(split)
            center_ds = CenterFilteredDataset(split)
            l = []
            for e in tqdm(expression_ds, desc=f'merging expressions {split}'):
                l.append(e)
            expressions = np.concatenate(l, axis=0)
            l = []
            for x in tqdm(center_ds, desc=f'merging centers {split}'):
                l.append(x)
            centers = np.concatenate(l, axis=0)
            assert len(expressions) == len(centers)
            f5[f'{split}/expressions'] = expressions
            f5[f'{split}/centers'] = centers


##
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


##
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

##
# recompute acculated features easily from cell tiles
ds = CellDataset('train')
e, o, m = ds[0]
x = o.transpose(2, 0, 1) * (m > 0)
ee = np.sum(x, axis=(1, 2))
ee /= m.sum()
ee = np.arcsinh(ee)
e_ds = ExpressionFilteredDataset('train')
# ee = e_ds.ds.scale(ee)
assert np.allclose(e, e_ds[0][0])
assert np.allclose(e, ee)

##
import torch
from torch.utils.data import DataLoader  # SequentialSampler

assert torch.cuda.is_available()

dataset = CellDataset('train')
loader = DataLoader(dataset, batch_size=1024, num_workers=16, pin_memory=True,
                    shuffle=True)

print(loader.__iter__().__next__())

##
quantiles_for_normalization = np.array([4.0549, 1.8684, 1.3117, 3.8141, 2.6172, 3.1571, 1.4984, 1.8866, 1.2621,
                                        3.7035, 3.6496, 1.8566, 2.5784, 0.9939, 1.4314, 2.1803, 1.8672, 1.6674,
                                        2.3555, 0.8917, 5.1779, 1.8002, 1.4042, 2.3873, 1.0509, 1.0892, 2.2708,
                                        3.4417, 1.8348, 1.8449, 2.8699, 2.2071, 1.0464, 2.5855, 2.0384, 4.8609,
                                        2.0277, 3.3281, 3.9273])


class PadByOne:
    def __call__(self, image):
        return F.pad(image, pad=[0, 1, 0, 1], mode='constant', value=0)


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class RGBCells(Dataset):
    def __init__(self, split, augment=False, aggressive_rotation=False):
        assert not (augment is False and aggressive_rotation is True)
        d = {'expression': False, 'center': False, 'ome': True, 'mask': True}
        self.ds = CellDataset(split, d)
        self.augment = augment
        self.aggressive_rotation = aggressive_rotation
        t = torchvision.transforms
        self.transform = t.Compose([
            t.ToTensor(),
            PadByOne()
        ])
        self.augment_transform = t.Compose([
            MyRotationTransform(angles=[90, 180, 270]) if not self.aggressive_rotation else t.RandomApply(
                nn.ModuleList([t.RandomRotation(degrees=360)]),
                p=0.6
            ),
            t.RandomHorizontalFlip(),
            t.RandomVerticalFlip(),
        ])
        # self.normalize = ome_normalization()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        x = self.ds[item][0]
        if len(self.ds[item]) == 2:
            mask = self.ds[item][1]
            mask = self.transform(mask)
            if self.augment:
                state = torch.get_rng_state()
                mask = self.augment_transform(mask)
            mask = mask.float()
        elif len(self.ds[item]) == 1:
            mask = None
        else:
            raise ValueError()
        x = self.transform(x)
        if self.augment:
            torch.set_rng_state(state)
            x = self.augment_transform(x)
        x = torch.asinh(x)
        # x = x[COOL_CHANNELS, :, :]
        x = x.permute(1, 2, 0)
        # x = (x - self.normalize.mean) / self.normalize.std
        x = x / quantiles_for_normalization
        x = x.permute(2, 0, 1)
        x = x.float()
        return x, mask


class PerturbedRGBCells(Dataset):
    def __init__(self, split: str, **kwargs):
        self.rgb_cells = RGBCells(split, **kwargs)
        self.seed = None
        # first element of the ds -> first elemnet of the tuple (=ome) -> shape[0]
        n_channels = self.rgb_cells[0][0].shape[0]
        self.corrupted_entries = torch.zeros((len(self.rgb_cells), n_channels), dtype=torch.bool)

    def perturb(self, seed=0):
        self.seed = seed
        from torch.distributions import Bernoulli
        dist = Bernoulli(probs=0.1)
        shape = self.corrupted_entries.shape
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.corrupted_entries = dist.sample(shape).bool()
        torch.set_rng_state(state)

    def __len__(self):
        return len(self.rgb_cells)

    def __getitem__(self, i):
        x, mask = self.rgb_cells[i]
        entries_to_corrupt = self.corrupted_entries[i, :]
        x[entries_to_corrupt] = 0.
        return x, mask, entries_to_corrupt


##

class PerturbedCellDatasetDebug(Dataset):
    def __init__(self, split):
        self.split = split
        # begin
        # let's just keep this class for debug purposes, to compare it with the one defined below
        self.ds = AccumulatedDataset(split, feature='mean', from_raw=True, transform=False)
        self.index_converter = FilteredMasksRelabeled(split).get_indices_conversion_arrays
        f = file_path(f'ah_filtered_untransformed_expression_tensor_merged_{split}.npy')
        # os.remove(f)
        if not os.path.isfile(f):
            all = []
            for i in tqdm(range(len(self.ds)), desc='merging expression tensor'):
                e = self.ds[i]
                new_e = ExpressionFilteredDataset.expression_old_to_new(e, i, index_converter=self.index_converter)
                all.append(new_e)
            merged = np.concatenate(all, axis=0)
            np.save(f, merged)
        self.merged = torch.tensor(np.load(f))
        # end
        self.merged = torch.asinh(self.merged)
        self.merged /= quantiles_for_normalization
        self.merged = self.merged.float()
        self.corrupted_entries = torch.zeros_like(self.merged, dtype=torch.bool)
        self.original_merged = None
        self.seed = None

    def perturb(self, seed=0):
        self.seed = seed
        from torch.distributions import Bernoulli
        dist = Bernoulli(probs=0.1)
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        shape = self.merged.shape
        self.corrupted_entries = dist.sample(shape).bool()
        torch.set_rng_state(state)
        self.original_merged = self.merged.clone()
        self.merged[self.corrupted_entries] = 0.

    def __len__(self):
        return len(self.merged)

    def __getitem__(self, i):
        return self.merged[i, :], self.corrupted_entries[i, :]


class PerturbedCellDataset(Dataset):
    def __init__(self, split: str):
        self.cell_dataset = CellDataset(split, features={'expression': True, 'center': False, 'ome': False,
                                                         'mask': False})
        self.seed = None
        # first element of the ds -> first elemnet of the tuple (=expression matrix) -> shape[0]
        n_channels = self.cell_dataset[0][0].shape[0]
        self.corrupted_entries = torch.zeros((len(self.cell_dataset), n_channels), dtype=torch.bool)

    def perturb(self, seed=0):
        self.seed = seed
        from torch.distributions import Bernoulli
        dist = Bernoulli(probs=0.1)
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        shape = self.corrupted_entries.shape
        self.corrupted_entries = dist.sample(shape).bool()
        torch.set_rng_state(state)

    def __len__(self):
        return len(self.cell_dataset)

    def __getitem__(self, i):
        x, mask = self.cell_dataset[i]
        entries_to_corrupt = self.corrupted_entries[i, :]
        x[entries_to_corrupt] = 0.
        return x, mask, entries_to_corrupt

ds0 = PerturbedCellDatasetDebug('train')
ds1 = PerturbedCellDatasetDebug('train')
assert ds0[42] == ds1[42]
##
print()
print('done')

    ##
    #
    # @staticmethod
    # def get_mean_and_std(ds):
    #     l = []
    #     for x in ds:
    #         l.append(x.numpy())
    #     z = np.concatenate(l, axis=0)
    #     mu = np.mean(z, axis=0)
    #     std = np.std(z, axis=0)
    #     return mu, std
    #
    # def scale(self, x):
    #     return (x - self.mu) / self.std
    #
    # def scale_back(self, z):
    #     return self.mu + self.std * z
    #
    # def __getitem__(self, i):
    #     x = self.accumulated_ds[i]
    #     z = self.scale(x)
    #     return z
