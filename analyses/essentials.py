import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from data2 import PerturbedCellDataset, file_path
from analyses.ab_images_vs_expression.ab_aa_expression_latent_samples import (
    nearest_neighbors,
    compute_knn_purity,
    compare_clusters,
    louvain_plot,
    precompute, compute_knn
)
from analyses.ac_third_party.ac_aa_scvi import scanpy_compute
from analyses.aa_reconstruction_benchmark.aa_ad_reconstruction import Space, Prediction, transform

def merge_perturbed_cell_dataset(ds: PerturbedCellDataset):
    loader = DataLoader(
        ds,
        batch_size=1024,
        num_workers=16,
    )
    l0 = []
    l1 = []
    for data in tqdm(loader, desc='merging cell ds'):
        expression, _, is_perturbed = data
        l0.append(expression)
        l1.append(is_perturbed)
    expressions = np.concatenate(l0, axis=0)
    are_perturbed = np.concatenate(l1, axis=0)
    return expressions, are_perturbed

def save_plot(name):
    f = file_path("plots")
    os.makedirs(f, exist_ok=True)
    plt.savefig(os.path.join(f, name))

LIGHT_GREEN = np.array([70, 155, 82]) / 255
DARK_GREEN = np.array([53, 121, 86]) / 255