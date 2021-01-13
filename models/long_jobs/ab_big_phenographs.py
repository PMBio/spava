## test
# python -m models.long_jobs.ab_big_phenographs --normalization-method raw --test

## test bsub
# bsub -q short -n 4 -M 8000 -R "rusage[mem=8000]" "python -m models.long_jobs.ab_big_phenographs --normalization-method raw --test"
# bsub -q short -n 4 -M 8000 -R "rusage[mem=8000]" "python -m models.long_jobs.ab_big_phenographs --normalization-method transformed --test"
# bsub -q short -n 4 -M 8000 -R "rusage[mem=8000]" "python -m models.long_jobs.ab_big_phenographs --normalization-method vae_mu --test"

## bsub (time: ?m, ?m, 15m)
# bsub -q verylong -n 32 -M 25000 -R "rusage[mem=25000]" "python -m models.long_jobs.ab_big_phenographs --normalization-method raw"
# bsub -q verylong -n 32 -M 25000 -R "rusage[mem=25000]" "python -m models.long_jobs.ab_big_phenographs --normalization-method transformed"
# bsub -q verylong -n 32 -M 25000 -R "rusage[mem=25000]" "python -m models.long_jobs.ab_big_phenographs --normalization-method vae_mu"
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

communities, graph, Q = phenograph.cluster(merged)

f0 = file_path(f'phenograph_{args.normalization_method}.hdf5')
f1 = file_path(f'phenograph_extra_{args.normalization_method}.pickle')
with h5py.File(f0, 'w') as f5:
    for o, begin, end in zip(index_info_omes, index_info_begins, index_info_ends):
        clustered = communities[begin:end]
        mu = merged[begin:end, :]
        f5[o + '/phenograph'] = clustered
        f5[o + '/mu'] = mu
pickle.dump({'communities': communities,
             'graph': graph,
             'Q': Q,
             'index_info_omes': index_info_omes,
             'index_info_begins': index_info_begins,
             'index_info_ends': index_info_ends}, open(f1, 'wb'))


