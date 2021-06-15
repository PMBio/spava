import os
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import h5py
from models.long_jobs.merge_cells import merge_cells
import phenograph
from ds import file_path
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
