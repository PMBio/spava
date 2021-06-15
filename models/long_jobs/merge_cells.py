import os
from tqdm import tqdm
import numpy as np
import argparse
import h5py
from ds import RawMeanDataset, TransformedMeanDataset, file_path
import pickle
from splits import train, validation, test

def merge_cells(which: str):
    assert which in ['raw', 'transformed', 'vae_mu']
    cells_list = []

    if which == 'raw' or which == 'transformed':
        if which == 'raw':
            ds_train = RawMeanDataset('train')
            ds_validation = RawMeanDataset('validation')
            # ds_test = RawMeanDataset('test')
        else:
            assert which == 'transformed'
            ds_train = TransformedMeanDataset('train')
            ds_validation = TransformedMeanDataset('validation')
            ds_test = TransformedMeanDataset('test')
        for x in tqdm(ds_train, desc='merging raw train'):
            cells_list.append(x.numpy())
        for x in tqdm(ds_validation, desc='merging raw validation'):
            cells_list.append(x.numpy())
        # for x in tqdm(ds_test, desc='merging raw test'):
        #     cells_list.append(x.numpy())
    else:
        assert which == 'vae_mu'
        model_path = file_path('vae_transformed_mean_dataset_LR_VB_S_0.0014685885989200848__3.8608662714605464e'
                               '-08__False')

        def the_path(instance, f):
            root = file_path(instance)
            assert os.path.isdir(root), root
            return os.path.join(root, f)

        # re embed for extra safety, not needed for newly trained models
        for split in ['train', 'validation']:
            from models.aa_train_vae_preprocessing import Vae
            # Vae = __import__('models.0_train_vae_preprocessing').__dict__['0_train_vae_preprocessing'].Vae
            import ignite.distributed as idist
            from h5_logger import H5Logger
            import torch

            ds_train = TransformedMeanDataset('train')
            ds_validation = TransformedMeanDataset('validation')

            n = 5
            model = Vae(in_channels=39, hidden_layer_dimensions=n, out_channels=39)
            model = model.to(idist.device())
            if not torch.cuda.is_available():
                model.load_state_dict(torch.load(the_path(model_path, 'model.torch'), map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(the_path(model_path, 'model.torch')))

            path = the_path(model_path, f'embedding_{split}.hdf5')
            if os.path.isfile(path):
                os.remove(path)
            if not os.path.isfile(path):
                EPOCH = 50
                print(f'embedding not found, creating it for EPOCH = {EPOCH}, path =', path)
                embedding_training_logger = H5Logger(path)
                embedding_training_logger.clear()
                if split == 'train':
                    ds = ds_train
                elif split == 'validation':
                    ds = ds_validation
                else:
                    raise ValueError()

                # only_first_5_samples = False
                with torch.no_grad():
                    list_of_reconstructed = []
                    list_of_mu = []
                    list_of_log_var = []
                    iterator = tqdm(ds, desc=f'embedding {split}', position=0, leave=True)
                    for i, data in enumerate(iterator):
                        data.to(idist.device())
                        data = torch.unsqueeze(data, 0)
                        # data = [torch.unsqueeze(x, 0) for x in data]
                        recon_batch, mu, log_var = model.forward_step(data, model)
                        ome_filename = ds.filenames[i]

                        def u(x):
                            if len(x.shape) == 3 and x.shape[0] == 1:
                                x = x.reshape(-1, x.shape[2])
                            return x

                        f = lambda x: x.cpu().detach().numpy()
                        # reconstructed = ome_dataset.scale_back(reconstructed.cpu().detach(), i)
                        a = f(u(recon_batch))
                        b = f(u(mu))
                        c = f(u(log_var))
                        data = {f'{ome_filename}/reconstructed': a,
                                f'{ome_filename}/mu': b,
                                f'{ome_filename}/log_var': c}
                        embedding_training_logger.log(EPOCH, data)

            # read the embedded data
            f = the_path(model_path, f'embedding_{split}.hdf5')
            with h5py.File(f, 'r') as f5:
                assert len(f5.keys()) == 1
                k, v = f5.items().__iter__().__next__()
                print(f'EPOCH {k}', flush=True)

                ds = RawMeanDataset(split)
                for o in tqdm(ds.filenames, desc='merging vae mu'):
                    cells_list.append(v[o]['mu'][...])

    all_cells = np.concatenate(cells_list, axis=0)
    cell_count = 669652
    assert all_cells.shape == (cell_count, 39) or all_cells.shape == (cell_count, 5), all_cells.shape

    # TODO:::::::::::::::::::::::::::::::::: to add also cells from validation set ;;;;;;;;;;;;;;;;;;;;;;;;;;;
    index_info_omes = []
    index_info_begins = []
    index_info_ends = []

    latest_end = 0
    for o, x in zip(ds_train.filenames, ds_train):
        begin = latest_end
        end = latest_end + len(x)
        latest_end += end - begin
        index_info_omes.append(o)
        index_info_begins.append(begin)
        index_info_ends.append(end)

    f = file_path('merged_cells_info.pickle')
    from atomicwrites import atomic_write
    # a check that the file that we replace is identical would not hurt
    with atomic_write(f, mode='wb', overwrite=True) as f:
        f.write(pickle.dumps((index_info_omes, index_info_begins, index_info_ends)))

    return all_cells


def ome_to_begin_end(ome):
    index_info_omes, index_info_begins, index_info_ends = pickle.load(open(file_path('merged_cells_info.pickle'), 'rb'))
    i = index_info_omes.index(ome)
    return index_info_begins[i], index_info_ends[i]


def cell_index_to_ome(cell_index):
    index_info_omes, index_info_begins, index_info_ends = pickle.load(open(file_path('merged_cells_info.pickle'), 'rb'))
    assert index_info_begins[0] == 0
    big_end = index_info_ends[-1]
    assert 0 <= cell_index < big_end
    index = None
    for i, a in enumerate(index_info_begins):
        if a > cell_index:
            index = i - 1
            break
    if index is None:
        index = len(index_info_begins) - 1
    assert index_info_begins[index] <= cell_index <= index_info_ends[index]
    return index_info_omes[index]


if __name__ == '__main__':
    merged = merge_cells('vae_mu')
    print(merged.shape)

    o = cell_index_to_ome(15000)
    print(o)
    begin, end = ome_to_begin_end(o)
    print(begin, end)
    print(cell_index_to_ome(begin - 1))
    print(cell_index_to_ome(begin))
    print(cell_index_to_ome(end - 1))
    print(cell_index_to_ome(end))
