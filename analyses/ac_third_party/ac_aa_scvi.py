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
    categorical_covariate_keys=["batch"],
)
a

##
# to navigate there with PyCharm and set a breakpoint on a warning
import scvi.core.distributions

model = scvi.model.SCVI(a)
##
# the following code, as it is, doesn't work
#     from data2 import file_path
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
model.train(train_size=1.0, n_epochs=10)

##
elbo = model.trainer.current_loss.item()

##
z = model.get_latent_representation()
a.shape
z.shape
b = ad.AnnData(z)
random_indices = np.random.choice(len(a), 10000, replace=False)
aa = a[random_indices]
bb = b[random_indices]

##
# plotting the raw data
sc.tl.pca(aa)
sc.pl.pca(aa, title='pca, raw data (sum)')
##
print('computing neighbors... ', end='')
sc.pp.neighbors(aa)
print('done')
print('computing umap... ', end='')
sc.tl.umap(aa)
print('done')
print('computing louvain... ', end='')
sc.tl.louvain(aa)
print('done')
##
sc.pl.umap(aa, color='louvain', title='umap with louvain, scvi latent (sum)')
##
# plotting the embedded data
sc.tl.pca(bb)
sc.pl.pca(bb, title='pca, raw data (sum)')
##
print('computing neighbors... ', end='')
sc.pp.neighbors(bb)
print('done')
print('computing umap... ', end='')
sc.tl.umap(bb)
print('done')
print('computing louvain... ', end='')
sc.tl.louvain(bb)
print('done')
##
sc.pl.umap(bb, color='louvain', title='umap with louvain, scvi latent (sum)')
##
from analyses.ab_images_vs_expression.ab_aa_expression_latent_samples import compare_clusters, nearest_neighbors, compute_knn
compare_clusters(aa, bb, description='"raw data (sum)" vs "scvi latent"')
compute_knn(aa)
compute_knn(bb)
nearest_neighbors(nn_from=aa, plot_onto=bb, title='nn from "raw data (sum)" to "scvi latent"')