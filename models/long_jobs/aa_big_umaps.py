## env stuff
# scp /data/l989o/deployed/spatial_uzh/install_env.sh l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/
# scp /data/l989o/deployed/spatial_uzh/requirements_cuda.yml l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/
# scp /data/l989o/deployed/spatial_uzh/pip.sh l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/

## data transfer
# rsync -avh /data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/phyper_data/accumulated_features/b3f06e1c82889221ec4ac7c901afe5295666b7ab905716bf32072ba1e2920abb/cell_features.hdf5 l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/data/spatial_uzh_processed/phyper_data/accumulated_features/b3f06e1c82889221ec4ac7c901afe5295666b7ab905716bf32072ba1e2920abb/cell_features.hdf5
# rsync -avh /data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/a/ok_cells_train.npy l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/data/spatial_uzh_processed/a/ok_cells_train.npy
# rsync -avh /data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/a/ok_cells_validation.npy l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/data/spatial_uzh_processed/a/ok_cells_validation.npy
# rsync -avh /data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/a/ok_cells_test.npy l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/data/spatial_uzh_processed/a/ok_cells_test.npy
# rsync -avh /data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/a/scaler_train.pickle l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/data/spatial_uzh_processed/a/scaler_train.pickle
# rsync -avh /data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/a/scaler_validation.pickle l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/data/spatial_uzh_processed/a/scaler_validation.pickle
# rsync -avh /data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/a/scaler_test.pickle l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/data/spatial_uzh_processed/a/scaler_test.pickle
# rsync -avh /data/l989o/deployed/spatial_uzh/data/spatial_uzh_processed/a/vae_transformed_mean_dataset_LR_VB_S_0.0014685885989200848__3.8608662714605464e-08__False l989o@odcf-lsf01.dkfz.de:/icgc/dkfzlsdf/analysis/B260/projects/spatial_a/data/spatial_uzh_processed/a/vae_transformed_mean_dataset_LR_VB_S_0.0014685885989200848__3.8608662714605464e-08__False

## test
# python -m models.long_jobs.aa_big_umaps --normalization-method raw --test

## test bsub
# bsub -q short -n 4 -M 8000 -R "rusage[mem=8000]" "python -m models.long_jobs.a --normalization-method raw --test"
# bsub -q short -n 4 -M 8000 -R "rusage[mem=8000]" "python -m models.long_jobs.aa_big_umaps --normalization-method transformed --test"
# bsub -q short -n 4 -M 8000 -R "rusage[mem=8000]" "python -m models.long_jobs.aa_big_umaps --normalization-method vae_mu --test"

## bsub
# bsub -q verylong -n 32 -M 25000 -R "rusage[mem=25000]" "python -m models.long_jobs.aa_big_umaps --normalization-method raw"
# bsub -q verylong -n 32 -M 25000 -R "rusage[mem=25000]" "python -m models.long_jobs.aa_big_umaps --normalization-method transformed"
# bsub -q verylong -n 32 -M 25000 -R "rusage[mem=25000]" "python -m models.long_jobs.aa_big_umaps --normalization-method vae_mu"
# ----------------------------------------------------------------------------------------------------

import os
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import h5py
from models.long_jobs.merge_cells import merge_cells
import phenograph
from data import file_path
import scanpy as sc
import anndata as ad
import time

parser = argparse.ArgumentParser()
parser.add_argument('--normalization-method', type=str, required=True)
parser.add_argument('--test', action='store_const', const=True)
args = parser.parse_args()

assert args.normalization_method in ['raw', 'transformed', 'vae_mu']

merged = merge_cells(args.normalization_method)
print(merged.shape)

from models.long_jobs.merge_cells import ome_to_begin_end, cell_index_to_ome

o = cell_index_to_ome(15000)
print(o)
begin, end = ome_to_begin_end(o)
print(begin, end)
print(cell_index_to_ome(begin - 1))
print(cell_index_to_ome(begin))
print(cell_index_to_ome(end - 1))
print(cell_index_to_ome(end))

index_info_omes, index_info_begins, index_info_ends = pickle.load(open(file_path('merged_cells_info.pickle'), 'rb'))

if args.test:
    small = index_info_ends[3]
    merged = merged[:small, :]

a = ad.AnnData(merged)

print('computing pca')
start = time.time()
sc.tl.pca(a, svd_solver='arpack')
print(f'pca computed: {time.time() - start}')

print('computing neighbors')
start = time.time()
sc.pp.neighbors(a)
print(f'neighbors computed: {time.time() - start}')

# sc.tl.leiden(b0)

print('computing umap')
start = time.time()
sc.tl.umap(a)
print(f'umap computed: {time.time() - start}')

f = file_path(f'umap_{args.normalization_method}.adata')
a.write(f)
# with h5py.File(f0, 'w') as f5:
#     for o, begin, end in zip(index_info_omes, index_info_begins, index_info_ends):
#         clustered = communities[begin:end]
#         mu = merged[begin:end, :]
#         f5[o + '/phenograph'] = clustered
#         f5[o + '/mu'] = mu
# pickle.dump({'communities': communities,
#              'graph': graph,
#              'Q': Q,
#              'index_info_omes': index_info_omes,
#              'index_info_begins': index_info_begins,
#              'index_info_ends': index_info_ends}, open(f1, 'wb'))
#
#
