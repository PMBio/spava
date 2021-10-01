##
from data2 import file_path
import pickle
from analyses.essentials import louvain_plot, scanpy_compute
from analyses.aa_reconstruction_benchmark.aa_ad_reconstruction import Prediction
import matplotlib.pyplot as plt
import numpy as np

##
d0 = pickle.load(open(file_path("scvi_scores.pickle"), "rb"))
d1 = pickle.load(open(file_path("ah_scores.pickle"), "rb"))
d2 = pickle.load(open(file_path('baseline_scores.pickle'), 'rb'))
d = {**d0, **d1, **d2}
for k, v in d.items():
    print(k)
    p = Prediction(**v)
    d[k] = v
    # print(np.argsort(v)[0])
    # print(np.argsort(v)[round(len(v) / 2)])
    # print(np.argsort(v)[-1])

##
adata0 = pickle.load(open(file_path('latent_anndata_from_scvi.pickle'), 'rb'))
adata1 = pickle.load(open(file_path('latent_anndata_from_ah_model.pickle'), 'rb'))

##
from utils import reproducible_random_choice
assert len(adata0) == len(adata1)
ii = reproducible_random_choice(len(adata0), 10000)
bdata0 = adata0[ii].copy()
bdata1 = adata1[ii].copy()

##
scanpy_compute(bdata0)
louvain_plot(bdata0, 'scVI latent')
scanpy_compute(bdata1)
louvain_plot(bdata1, 'vanilla VAE latent')