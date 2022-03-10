##
import shutil

import pandas as pd
import spatialmuon as smu

import matplotlib.pyplot as plt
import matplotlib

import os

from utils import get_execute_function, file_path
import colorama
import scanpy as sc
import h5py

# os.environ['SPATIALMUON_TEST'] = 'aaa'
# os.environ['SPATIALMUON_NOTEBOOK'] = 'aaa'
e_ = get_execute_function()
# matplotlib.use('module://backend_interagg')

plt.style.use("dark_background")

##
def get_smu_file(initialize=False):
    "/data/spatialmuon/datasets/visium_mousebrain/smu"
    RAW_FOLDER = file_path('spatialmuon/visium_mousebrain')
    PROCESSED_FOLDER = file_path("spatialmuon_processed/visium_mousebrain")

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    if 'SPATIALMUON_TEST' not in os.environ:
        src_f = os.path.join(RAW_FOLDER, "visium.h5smu")
        des_f = os.path.join(PROCESSED_FOLDER, "visium.h5smu")
    else:
        src_f = os.path.join(RAW_FOLDER, "small_visium.h5smu")
        des_f = os.path.join(PROCESSED_FOLDER, "small_visium.h5smu")

    if initialize:
        shutil.copy(src_f, des_f)

    s = smu.SpatialMuData(des_f)
    return s

##
if e_():
    s = get_smu_file(initialize=True)
    if 'SPATIALMUON_TEST' not in os.environ:
        name = "ST8059049"
        name_hne = name + "H&E"

        regions = s["Visium"][name]
        raster = s["Visium"][name_hne]
    else:
        regions = s['visium']['expression']
        raster = s['visium']['image']
##
if e_():
    _, ax = plt.subplots(1)
    raster.plot(ax=ax)
    regions.plot(0, ax=ax)
    plt.show()

##
if e_():
    # biologically relevant channels
    if 'SPATIALMUON_TEST' not in os.environ:
        chosen_genes = ['Olfm1', 'Plp1', 'Itpka']
    else:
        chosen_genes = ["Prex2", "Atp6v1h", "Xkr4"]
    regions.plot(channels=chosen_genes)
    raster.plot()

##
if e_():
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
if e_():
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    adata

##
if e_():
    plt.style.use("default")
    sc.pl.violin(adata, ["n_genes_by_counts", "total_counts"], jitter=0.4, multi_panel=True)
    plt.style.use("dark_background")
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")

##
if e_():
    # we need statsmodels 0.13.2 for regress_out
    # https://stackoverflow.com/questions/71106940/cannot-import-name-centered-from-scipy-signal-signaltools
    # sc.pp.regress_out(adata, ["total_counts"])
    sc.pp.scale(adata, max_value=10)

##
# if I use this I get a segfault from protobuf for some reasons, let us make neighbors call implicitly a version of
# pca that doesn't trigger this
if False:
    sc.pp.pca(adata)
    sc.pl.pca(adata, color=chosen_genes)
    plt.style.use("default")
    sc.pl.pca_variance_ratio(adata, log=True)
    plt.style.use("dark_background")

##
if e_():
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

if e_():
    sc.pl.umap(adata, color=chosen_genes)

##
if e_():
    sc.tl.leiden(adata)

if e_():
    sc.pl.umap(adata, color=['leiden'])
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
if e_():
    processed_regions = smu.Regions(X=adata.X, var=adata.var, masks=regions.masks.clone(), anchor=regions.anchor)
    processed_regions.var.reset_index(inplace=True)
    processed_regions.var.rename(columns={'index': 'channel_name'})
    processed_regions.masks.obs['leiden'] = adata.obs['leiden'].to_numpy()
    processed_regions.masks.obs['leiden'] = processed_regions.masks.obs['leiden'].astype('category')
    processed_regions.masks.set_all_has_changed(new_value=True)
    processed_regions.plot(chosen_genes)
    processed_regions.masks.plot('leiden')

    if 'SPATIALMUON_TEST' not in os.environ:
        if 'processed' in s['Visium']:
            del s['Visium']['processed']
        s['Visium']['processed'] = processed_regions
    else:
        if 'processed' in s['visium']:
            del s['visium']['processed']
        s['visium']['processed'] = processed_regions
    s.commit_changes_on_disk()
    t = get_smu_file()
    print(t)
    print('ooo')

##
if e_():
    s = get_smu_file()
    if 'SPATIALMUON_TEST' not in os.environ:
        m0 = s['Visium']['ST8059049'].masks
        m1 = s['Visium']['processed'].masks
    else:
        m0 = s['visium']['expression'].masks
        m1 = s['visium']['processed'].masks

    m0.plot()
    m1.plot()
##
if e_():
    s = get_smu_file()
    if 'SPATIALMUON_TEST' not in os.environ:
        processed_regions = s['Visium']['processed']
        raster = s['Visium']['ST8059049H&E']
    else:
        processed_regions = s['visium']['processed']
        raster = s['visium']['image']
    if 'SPATIALMUON_TEST' not in os.environ:
        bb = smu.BoundingBox(x0=250, x1=1000, y0=250, y1=750)
        transformed_bb = processed_regions.anchor.transform_bounding_box(bb)
    else:
        transformed_bb = None


    plt.figure()
    ax = plt.gca()
    processed_regions.masks.plot(fill_colors=None, outline_colors='leiden', ax=ax, bounding_box=transformed_bb)
    raster.plot(ax=ax, show_legend=False, bounding_box=transformed_bb)
    raster.set_lims_to_bounding_box(transformed_bb)
    plt.show()
##
if e_():
    s = get_smu_file()
    if 'SPATIALMUON_TEST' not in os.environ:
        masks = s['Visium']['processed'].masks
        raster = s['Visium']['ST8059049H&E']
    else:
        masks = s['visium']['processed'].masks
        raster = s['visium']['image']
    print(f'{colorama.Fore.MAGENTA}extracting tiles{colorama.Fore.RESET}')
    f = file_path("visium_tiles.hdf5")
    with h5py.File(f, "w") as f5:
        tiles = smu.Tiles(
            raster=raster,
            masks=masks,
            tile_dim_in_pixels=32,
        )
        filename = os.path.basename(s.backing.filename)
        f5[f"{filename}/raster"] = tiles.tiles

##
print('done')
