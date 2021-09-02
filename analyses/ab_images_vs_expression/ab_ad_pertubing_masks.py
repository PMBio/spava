##
import matplotlib.cm
import numpy as np
import time
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import skimage.measure

from models.ag_conv_vae_lightning import PerturbedRGBCells
from models.ah_expression_vaes_lightning import PerturbedCellDataset

ds = PerturbedRGBCells(split="validation")

cells_ds = PerturbedCellDataset(split="validation")
if False:
    ds.perturb()
    cells_ds.perturb()

assert torch.all(ds.corrupted_entries == cells_ds.corrupted_entries)

##
assert len(ds) == len(cells_ds)
n = len(ds)
for i in tqdm(range(n)):
    ome, mask, _ = ds[i]
    o = ome * mask
    e = o.mean(dim=(1, 2))
    expression, _ = cells_ds[i]
    assert torch.allclose(expression, e)

##

models = {
    "resnet_vae": "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_7/checkpoints"
                  "/epoch=3-step=1610.ckpt",
}

rgb_ds = ds.rgb_cells
from models.ag_conv_vae_lightning import VAE as ResNetVAE

the_model = "resnet_vae"
# the_model = 'resnet_vae_last_channel'
resnet_vae = ResNetVAE.load_from_checkpoint(models[the_model])
loader = DataLoader(rgb_ds, batch_size=16, num_workers=8, pin_memory=True)
data = loader.__iter__().__next__()

##
if False:
    start = time.time()
    list_of_z = []
    with torch.no_grad():
        for data in tqdm(loader, desc="embedding the whole validation set"):
            data = [d.to(resnet_vae.device) for d in data]
            z = [zz.cpu() for zz in resnet_vae(*data)]
            list_of_z.append(z)
    print(f"forwarning the data to the resnets: {time.time() - start}")

    torch.cuda.empty_cache()

##
from data2 import file_path

f = file_path("image_features.npy")
if False:
    mus = torch.cat([zz[2] for zz in list_of_z], dim=0).numpy()
    np.save(f, mus)
mus = np.load(f)
##
import scanpy as sc
import anndata as ad

a = ad.AnnData(mus)
sc.tl.pca(a)
sc.pl.pca(a)
##
random_indices = np.random.choice(len(a), 10000, replace=False)
b = a[random_indices]
##
print("computing umap... ", end="")
sc.pp.neighbors(b)
sc.tl.umap(b)
sc.tl.louvain(b)
print("done")
##
plt.figure()
l = b.obs["louvain"].tolist()
colors = list(map(int, l))
plt.scatter(
    b.obsm["X_umap"][:, 0],
    b.obsm["X_umap"][:, 1],
    s=1,
    c=colors,
    cmap=matplotlib.cm.tab20,
)
# plt.xlim([10, 20])
# plt.ylim([0, 10])
plt.show()
