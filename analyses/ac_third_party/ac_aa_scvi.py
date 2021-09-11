##
import scvi
import scanpy as sc
from torch.utils.data import DataLoader

from data2 import SumFilteredDataset
import numpy as np
from tqdm import tqdm
import anndata as ad
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
import os
import matplotlib.pyplot as plt

##
if False:
    # proxy for the DKFZ network
    # https://stackoverflow.com/questions/34576665/setting-proxy-to-urllib-request-python3
    os.environ["HTTP_PROXY"] = "http://193.174.53.86:80"
    os.environ["HTTPS_PROXY"] = "https://193.174.53.86:80"

##
if False:
    # to have a look at an existing dataset
    import scvi.data

    data = scvi.data.pbmc_dataset()
    data

##
ds = SumFilteredDataset("train")
l0 = []
l1 = []
for i, x in enumerate(tqdm(ds, "merging")):
    l0.append(x)
    l1.extend([i] * len(x))
raw = np.concatenate(l0, axis=0)
donor = np.array(l1)
a = ad.AnnData(raw)

##
s = pd.Series(donor, index=a.obs.index)
a.obs["batch"] = s

##
scvi.data.setup_anndata(
    a,
    # this is probably meaningless (if not even penalizing) for unseen data as the batches are different
    # categorical_covariate_keys=["batch"],
)
a

##
# TRAIN = True
TRAIN = False
if TRAIN:
    # to navigate there with PyCharm and set a breakpoint on a warning
    import scvi.core.distributions

    model = scvi.model.SCVI(a)

##
from data2 import file_path
# the following code, as it is, doesn't work
#     logger = TensorBoardLogger(save_dir=file_path("checkpoints"), name="scvi")
#     BATCH_SIZE = 128
#     indices = np.random.choice(len(a), BATCH_SIZE * 20, replace=False)
#
#     train_loader_batch = DataLoader(
#         a.X[indices, :],
#         batch_size=BATCH_SIZE,
#         num_workers=4,
#         pin_memory=True,
#     )
#     model.train(train_size=1., logger=logger, val_dataloaders=train_loader_batch)
#     model.__dict__
if TRAIN:
    model.train(train_size=1.0, n_epochs=10)
    model.save(file_path('scvi_model.scvi'))
else:
    model = scvi.model.SCVI.load(file_path('scvi_model.scvi'), adata=a)

##
if TRAIN:
    elbo = model.trainer.current_loss.item()

##
z = model.get_latent_representation()
a.shape
z.shape
b = ad.AnnData(z)
random_indices = np.random.choice(len(a), 10000, replace=False)
aa = a[random_indices]
bb = b[random_indices]

def scanpy_compute(an: ad.AnnData):
    sc.tl.pca(an)
    print("computing neighbors... ", end="")
    sc.pp.neighbors(an)
    print("done")
    print("computing umap... ", end="")
    sc.tl.umap(an)
    print("done")
    print("computing louvain... ", end="")
    sc.tl.louvain(an)
    print("done")
##
scanpy_compute(aa)
sc.pl.pca(aa, title="pca, raw data (sum)")
sc.pl.umap(aa, color="louvain", title="umap with louvain, scvi latent (sum)")

##
scanpy_compute(bb)
sc.pl.pca(bb, title="pca, raw data (sum)")
sc.pl.umap(bb, color="louvain", title="umap with louvain, scvi latent (sum)")

##
from analyses.ab_images_vs_expression.ab_aa_expression_latent_samples import (
    compare_clusters,
    nearest_neighbors,
    compute_knn,
)

compare_clusters(aa, bb, description='"raw data (sum)" vs "scvi latent"')
compute_knn(aa)
compute_knn(bb)
nearest_neighbors(
    nn_from=aa, plot_onto=bb, title='nn from "raw data (sum)" to "scvi latent"'
)

##
ds = SumFilteredDataset("validation")
l0 = []
l1 = []
for i, x in enumerate(tqdm(ds, "merging")):
    l0.append(x)
    l1.extend([i] * len(x))
raw = np.concatenate(l0, axis=0)
donor = np.array(l1)
a_val = ad.AnnData(raw)

##
# note that here with are embedding without the batch information; if you want to look at batches it does not make
# sense to use another set except to the training one, since the train/val/test split is done by patient first
scvi.data.setup_anndata(
    a_val,
)
z_val = model.get_latent_representation(a_val)
b_val = ad.AnnData(z_val)
random_indices_val = np.random.choice(len(a_val), 10000, replace=False)
aa_val = a_val[random_indices_val]
bb_val = b_val[random_indices_val]

##
scanpy_compute(aa_val)
scanpy_compute(bb_val)

##
sc.pl.pca(aa_val, title="pca, raw data (sum); validation set")
sc.pl.umap(aa_val, color="louvain", title="umap with louvain, scvi latent (sum); valiation set")
sc.pl.pca(bb_val, title="pca, raw data (sum); valiation set")
sc.pl.umap(bb_val, color="louvain", title="umap with louvain, scvi latent (sum); valiation set")

##
merged = ad.AnnData.concatenate(bb, bb_val, batch_categories=['train', 'validation'])
scanpy_compute(merged)
plt.figure()
ax = plt.gca()
sc.pl.umap(merged, color='batch', ax=ax, show=False)
plt.tight_layout()
plt.show()

##
size_factors = model.get_latent_library_size(a_val)

##
from data2 import AreaFilteredDataset
area_ds = AreaFilteredDataset('validation')

l = []
for x in tqdm(area_ds, desc='merging'):
    l.append(x)
areas = np.concatenate(l, axis=0)

##
from scipy.stats import pearsonr
print(size_factors.shape)
print(areas.shape)
r, p = pearsonr(size_factors.ravel(), areas.ravel())
plt.figure()
plt.scatter(size_factors, areas, s=0.5)
plt.xlabel('latent size factors')
plt.ylabel('cell area')
plt.title(f'r: {r:0.2f} (p: {p:0.2f})')
plt.show()

##
