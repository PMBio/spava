##
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import skimage
import skimage.io
import torch
import cv2
import vigra
# from tqdm.notebook import tqdm
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

COMPUTE = False
if __name__ == '__main__':
    PLOT = True
    # comment the next line unless you want to recompute the datasets
    COMPUTE = True
else:
    PLOT = False

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


if COMPUTE and False:
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
if PLOT:
    def plot_expression_dataset(ds):
        l = []
        for e in ds:
            l.append(e.mean(dim=0, keepdim=True))
        ee = torch.cat(l, dim=0)
        print(ee.shape)
        plt.figure()
        plt.imshow(ee.numpy())
        plt.show()


    ds = AccumulatedDataset('validation', 'mean', from_raw=True, transform=True)
    plot_expression_dataset(ds)


##

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


if PLOT:
    ds = ScaledAccumulatedDataset('validation', feature='mean', from_raw=True, transform=True)
    plot_expression_dataset(ds)


##
def get_ok_size_cells(split):
    areas_ds = AccumulatedDataset(split, 'count', from_raw=True, transform=False)
    all_areas = []
    # note that we remove the background before concatenating
    check = 0
    for area in tqdm(areas_ds, desc=f'filtering cells by areas, {split} set'):
        all_areas.append(area)
        check += len(area)

    areas = torch.cat(all_areas)
    assert len(areas) == check
    print(len(areas), 'cells in', split, 'set')

    min_area = 15
    max_area = 400
    if PLOT:
        plt.figure()
        plt.hist(areas.numpy(), bins=100)
        plt.axvline(x=min_area, c='r')
        plt.axvline(x=max_area, c='r')
        plt.xlabel('cell area')
        plt.ylabel('count')
        plt.title(f'distribution of cell area, {split} set')
        plt.show()

    ok_size_cells = (areas >= min_area) & (areas <= max_area)
    return ok_size_cells.numpy()


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
        new_t = torch.cat((torch.zeros_like(t)[:1], t))
        new_t[0] = background_value
    return new_t


if PLOT:
    print(prepend_background(np.array([3, 4, 5])))
    print(prepend_background(torch.tensor([3.2, 3.4]), background_value=-2))
##

if PLOT:
    split = 'train'
    ds = ScaledAccumulatedDataset(split, 'mean', from_raw=True, transform=True)
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
        self.ds = ScaledAccumulatedDataset(split, feature='mean', from_raw=True, transform=True)
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
########################################## REFACTORED ONLY BEFORE THIS POINT

DEBUG = False

# %%

# %%script false --no-raise-error

from scipy.ndimage import center_of_mass

f_in = file_path('accumulated_features/raw_accumulated.hdf5')
f_out = file_path('filtered_cells_dataset.hdf5')

DEBUG_WITH_PLOTS = False
if DEBUG_WITH_PLOTS:
    DEBUG = True
if DEBUG:
    lists_of_centers = {'train': [], 'validation': [], 'test': []}
    lists_of_expressions = {'train': [], 'validation': [], 'test': []}

with h5py.File(f_out, 'w') as f5_out:
    with h5py.File(f_in, 'r') as f5_in:
        for split in tqdm(['train', 'validation', 'test'], desc=f'split'):
            k = 0
            ds = ExpressionDataset(split)
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
                    # - 1 because in the hdf5 file the background has been removed
                    new = cell_index + 1
                    old = new_to_old[new].item()
                    center = f5_in[f'/{o}/region_center'][old - 1, :]
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


                    src0_a, src0_b, src1_a, src1_b, des0_a, des0_b, des1_a, des1_b = get_coords_for_padding(r, y.shape,
                                                                                                            y_center)

                    square_ome[des0_a:des0_b, des1_a:des1_b, :] = ome[a1:b1, a0:b0, :][src0_a:src0_b, src1_a:src1_b, :]
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
                            assert False


# %%

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


# %%

from sklearn.decomposition import PCA

if DEBUG:
    with h5py.File(f_out, 'r') as f5:
        ds = ExpressionDataset('train')
        l = []
        for e in tqdm(ds):
            l.append(e)
        e = np.concatenate(l, axis=0)
        len(e)
        ee = e[:k, ]
        llll = []
        for lll in tqdm(lists_of_expressions['train']):
            llll.append(lll.reshape((1, -1)))
        eee = np.concatenate(llll, axis=0)
        index_info = IndexInfo('train')
        assert np.all(ee == eee)
        print('expressions computed correctly')

        ds = CentersFilteredDataset('train')
        l = []
        for e in tqdm(ds):
            l.append(e)
        e = np.concatenate(l, axis=0)
        len(e)
        ee = e[:k, ]
        llll = []
        for lll in tqdm(lists_of_centers['train']):
            llll.append(lll.reshape((1, -1)))
        eee = np.concatenate(llll, axis=0)
        index_info = IndexInfo('train')
        assert np.all(ee == eee)
        print('region centers computed correctly')

# %%

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

# %%

import matplotlib

with h5py.File(file_path('filtered_cells_dataset.hdf5'), 'r') as f5:
    cell_k = 10000
    x = f5[f'train/omes/{cell_k}'][...]
    plt.imshow(f5[f'train/masks/{cell_k}'][...])
    axes = plt.subplots(8, 5, figsize=(5 * 2, 8 * 1.8))[1].flatten()
    for i in range(39):
        axes[i].imshow(x[:, :, i], cmap=matplotlib.cm.get_cmap('gray'))

# %%

% % script
false - -no -
raise -error

with h5py.File(file_path('merged_filtered_centers_and_expressions.hdf5'), 'w') as f5:
    for split in ['train', 'validation', 'test']:
        filenames = get_split(split)
        expression_ds = ExpressionDataset(split)
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


# %%

class CellDataset(Dataset):
    def __init__(self, split):
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
        return self.expressions[i], self.centers[i], self.f5_omes[f'{i}'][...], self.f5_masks[f'{i}'][...]


# %%

# %%script false --no-raise-error
ds = CellDataset('train')
h = []
w = []
for x in tqdm(ds):
    pass
#    h.append(x[3].shape[0])
#    w.append(x[3].shape[1])

hh = np.array(h)
ww = np.array(w)

plt.hist(hh)
plt.show()
plt.hist(ww)
plt.show()

# %%

import torch
from torch.utils.data import DataLoader  # SequentialSampler

# import ignite.distributed as idist
# idist.auto_dataloader

assert torch.cuda.is_available()
# def my_collate(batch):
#     expressions = torch.stack([torch.from_numpy(b[0]) for b in batch], 0)
#     centers = torch.stack([torch.from_numpy(b[1]) for b in batch], 0)
#     h = max([b[3].shape[0] for b in batch])
#     w = max([b[3].shape[1] for b in batch])
#     c = batch[0][2].shape[2]
#     n = len(batch)
#     omes = np.zeros((n, c, h, w))
#     masks = np.zeros((n, h, w))
#     for i in range(n):
#         b = batch[i]
#         ome = torch.from_numpy(batch[i][2]).permute(2, 0, 1)
#         small_h, small_w = ome.shape[1:]
#         omes[i, :, :small_h, :small_w] = ome
#         m = torch.from_numpy(batch[i][3])
#         masks[i, :small_h, :small_w] = m
#     return (expressions, centers, omes, masks)

dataset = CellDataset('train')
loader = DataLoader(dataset, batch_size=1024, num_workers=16, pin_memory=True,
                    shuffle=True)  # , collate_fn=my_collate) # , sampler=sampler,

# %%

% % time
loader.__iter__().__next__()
print()

# %%

n = 0
for x in tqdm(loader):
    pass

# %%

for x in ds:
    pass

# %%

n

# %%

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


# a simple custom collate function, just to show the idea
def my_collate(batch):
    print(batch)
    assert False
    #     data = [item[0] for item in batch]
    #     target = [item[1] for item in batch]
    #     target = torch.LongTensor(target)
    return [data, target]


def show_image_batch(img_list, title=None):
    num = len(img_list)
    fig = plt.figure()
    for i in range(num):
        ax = fig.add_subplot(1, num, i + 1)
        ax.imshow(img_list[i].numpy().transpose([1, 2, 0]))
        ax.set_title(title[i])

    plt.show()


#  do not do randomCrop to show that the custom collate_fn can handle images of different size
train_transforms = transforms.Compose([transforms.Scale(size=224),
                                       transforms.ToTensor(),
                                       ])

# change root to valid dir in your system, see ImageFolder documentation for more info
train_dataset = datasets.ImageFolder(root="/hd1/jdhao/toyset",
                                     transform=train_transforms)

trainset = DataLoader(dataset=train_dataset,
                      batch_size=4,
                      shuffle=True,
                      collate_fn=my_collate,  # use custom collate function here
                      pin_memory=True)

trainiter = iter(trainset)
imgs, labels = trainiter.next()

# print(type(imgs), type(labels))
show_image_batch(imgs, title=[train_dataset.classes[x] for x in labels])


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
