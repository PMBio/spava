##
import numpy as np
import scanpy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import Dataset

m = False
m = __name__ == '__main__'

##
N_CHANNELS = 39
PIXELS = 32
SPARSITY = 1000
# SPARSITY = 50

##
adata = sc.read_10x_mtx(
    "/data/l989o/data/scrna_seq/pbmc3k/filtered_gene_bc_matrices/hg19/",  # the directory with the `.mtx` file
    var_names="gene_symbols",  # use gene symbols for the variable names (variables-axis index)
    cache=True,
)
adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`

##
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
# adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
# sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# sc.pp.calculate_qc_metrics(adata)
# adata = adata[adata.obs.n_genes_by_counts < 2500, :]
# adata = adata[adata.obs.pct_counts_mt < 5, :]
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)

##
if m:
    # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # sc.pl.highly_variable_genes(adata)
    # adata = adata[:, adata.var.highly_variable]
    # sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pl.pca(adata, color="CST3")
    sc.pl.pca_variance_ratio(adata, log=True)

    ##
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    ##
    sc.tl.umap(adata)

    ##
    sc.tl.louvain(adata)

    ##
    sc.pl.umap(adata, color=["louvain"], show=False, title="PBMC3k, Louvain clusters")
    ax = plt.gca()
    plt.tight_layout()
    plt.show()

##
x = adata.X.todense().A
hvg = x.std(axis=0).argsort()[::-1][:1000]
state = np.random.get_state()
np.random.seed(42)
hvg = np.random.choice(hvg, N_CHANNELS, replace=False)
np.random.set_state(state)

x_hvg = x[:, hvg]

##
if m:
    plt.figure()
    plt.scatter(x_hvg.mean(axis=0), x_hvg.std(axis=0))
    plt.show()

##
filter0 = np.zeros((PIXELS, PIXELS))
for i in range(PIXELS):
    for j in range(PIXELS):
        c = PIXELS // 2
        filter0[i, j] = (i - c) ** 2 + (j - c) ** 2
filter0 = filter0 / filter0.sum() * SPARSITY
filter1 = filter0.max() - filter0
filter1 = filter1 / filter1.sum() * SPARSITY
filter1 = filter1 * filter0.max() / filter1.max()
filter2 = np.random.rand(PIXELS, PIXELS) * SPARSITY
filter2 = filter2 * filter0.max() / filter2.max()

if m:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(filter0)
    plt.colorbar()
    plt.title("expression outside the nucleus")

    plt.subplot(1, 3, 2)
    plt.imshow(filter1)
    plt.colorbar()
    plt.title("expression inside the nucleus")

    plt.subplot(1, 3, 3)
    plt.imshow(filter2)
    plt.colorbar()
    plt.title("no spatial pattern")

    plt.show()

##
if m:
    plt.figure()
    plt.hist(x_hvg.flatten())
    plt.show()

##
gene_specific_dispersions = np.random.rand(N_CHANNELS)

##
cells = []
filter = [0] * 1000 + [1] * 1000 + [2] * (len(x_hvg) - 2000)
random.Random(42).shuffle(filter)
for cell in tqdm(range(x_hvg.shape[0])):
    genes = []
    for gene in range(x_hvg.shape[1]):
        e = x_hvg[cell, gene]
        if filter[cell] == 0:
            f = filter0 * e
        elif filter[cell] == 1:
            f = filter1 * e
        else:
            f = filter2 * e
        k = np.random.poisson(f)
        genes.append(k)
    cells.append(genes)
t = np.array(cells)

##
if m:
    axes = plt.subplots(5, 3, figsize=(9, 15))[1].flatten()
    pattern = ["cytoplasm", "nucleus", "no pattern"]
    c = 0
    for i in range(len(axes)):
        ax = axes[i]
        i = i + 100
        ax.imshow(t[i, c])
        ax.set(title=pattern[filter[i]])
    plt.tight_layout()
    plt.show()

###
class FakeRGBCells(Dataset):
    def __init__(self, split: str):
        splits = {'train': slice(0, 2000),
                  'validation': slice(2000, 2800),}
                  # 'test': slice(2800, 2800)}
        s = splits[split]
        self.t = torch.tensor(t[s])
        self.filter = filter[s]

    def __len__(self):
        return len(self.t)

    def __getitem__(self, item):
        x = self.t[item, ...].float()
        mask = torch.ones_like(x, dtype=torch.float)[:1, :, :]
        corrupted_entries = torch.zeros_like(x, dtype=torch.bool)[:, 0, 0]
        return x, mask, corrupted_entries

if m:
    ds = FakeRGBCells('train')
    print(ds[0])


