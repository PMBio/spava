##
import pandas as pd
import spatialmuon as smu

import matplotlib.pyplot as plt

import os

from utils import setup_ci, file_path
import colorama
import scanpy as sc
import h5py

c_, p_, t_, n_ = setup_ci(__name__)

plt.style.use("dark_background")

##
RAW_FOLDER = "/data/spatialmuon/datasets/visium_mousebrain/smu"
PROCESSED_FOLDER = file_path("spatialmuon_processed")

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

##
f = os.path.join(RAW_FOLDER, "visium.h5smu")
s = smu.SpatialMuData(f)

##
name = "ST8059049"
name_hne = name + "H&E"

regions = s["Visium"][name]
raster = s["Visium"][name_hne]

##
# regions.plot(channels=["Olfm1", "Plp1", "Itpka", 0])
# raster.plot()

##
# x = regions.X[...]
# if isinstance(x, ad._core.sparse_dataset.SparseDataset):
#     x = x.todense()
regions.masks.obs["region_center_x"] = regions.masks._masks_centers[:, 0]
regions.masks.obs["region_center_y"] = regions.masks._masks_centers[:, 1]
adata = smu.Converter().regions_to_anndata(regions)
adata.var.index = regions.var["channel_name"]
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=3)

##
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
adata

##
plt.style.use("default")
sc.pl.violin(adata, ["n_genes_by_counts", "total_counts"], jitter=0.4, multi_panel=True)
plt.style.use("dark_background")
sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")

##
sc.pp.regress_out(adata, ["total_counts"])
sc.pp.scale(adata, max_value=10)

##
# if I use this I get a segfault from protobuf for some reasons, let us make neighbors call implicitly a version of
# pca that doesn't trigger this
if False:
    sc.pp.pca(adata)
    sc.pl.pca(adata, color=["Olfm1", "Plp1", "Itpka"])
    plt.style.use("default")
    sc.pl.pca_variance_ratio(adata, log=True)
    plt.style.use("dark_background")

##
sc.pp.neighbors(adata)

##
sc.tl.umap(adata)

##
sc.pl.umap(adata, color=["Olfm1", "Plp1", "Itpka"])

##
sc.tl.leiden(adata)

##
sc.pl.umap(adata, color=['leiden'])

##
sc.pl.spatial(adata, color=['leiden'], spot_size=20)

##
if False:
    # to understand that the data has not been log(1 + x) transformed
    x = np.asarray(adata.X.todense())
    x = np.log1p(x)
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    plt.figure()
    plt.scatter(mu, std)
    plt.xlabel("mu")
    plt.ylabel("std")
    plt.show()
# regions.var

##
processed_regions = smu.Regions(X=adata.X, var=adata.var, masks=regions.masks)
processed_regions.var.reset_index(inplace=True)
processed_regions.var.rename(columns={'index': 'channel_name'})
processed_regions.masks.obs['leiden'] = adata.obs['leiden'].to_numpy()
processed_regions.masks.obs['leiden'] = processed_regions.masks.obs['leiden'].astype('category')
processed_regions.plot(["Olfm1", "Plp1", "Itpka"])
processed_regions.masks.plot('leiden')

##
plt.figure()
ax = plt.gca()
processed_regions.masks.plot(fill_colors=None, outline_colors='leiden', ax=ax)
raster.plot(ax=ax, show_legend=False)
ax.set(xlim=(250, 1000), ylim=(250, 750))
plt.show()
##
f = os.path.join(RAW_FOLDER, "visium.h5smu")
s = smu.SpatialMuData(f)
if 'processed' in s['Visium']:
    del s['Visium']['processed']
s['Visium']['processed'] = processed_regions

##

if n_ or t_ or c_:
    print(f'{colorama.Fore.MAGENTA}extracting tiles{colorama.Fore.RESET}')
    f = file_path("visium_tiles.hdf5")
    with h5py.File(f, "w") as f5:
        tiles = smu.Tiles(
            raster=raster,
            masks=processed_regions.masks,
            tile_dim=32,
        )
        filename = os.path.basename(s.backing.filename)
        f5[f"{filename}/raster"] = tiles.raster_tiles
        f5[f"{filename}/masks"] = tiles.masks_tiles

pass
