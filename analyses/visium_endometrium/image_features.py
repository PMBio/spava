##

import os
import shutil
import tempfile

import colorama
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import optuna
import spatialmuon
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from analyses.analisys_utils import louvain_plot, scanpy_compute
from analyses.imputation_score import Prediction
from datasets.loaders.visium_endometrium_loaders import (
    get_cells_data_loader,
    CellsDataset,
    get_data_per_sample,
)
from datasets.visium_endometrium import get_smu_file, visium_endrometrium_samples
from utils import get_execute_function, parse_flags

e_ = get_execute_function()

##
# flags = parse_flags(default={"MODEL_NAME": "expression_vae"})
flags = parse_flags(default={"MODEL_NAME": "image_expression_conv_vae", 'TILE_SIZE': 'large'})
MODEL_NAME = flags["MODEL_NAME"]
is_expression_vae = MODEL_NAME == "expression_vae"
is_image_expression_conv_vae = MODEL_NAME == "image_expression_conv_vae"

##
if is_expression_vae:
    MODEL_FULLNAME = f"visium_endometrium_{MODEL_NAME}"
elif is_image_expression_conv_vae:
    TILE_SIZE = flags["TILE_SIZE"]
    MODEL_FULLNAME = f"visium_endometrium_{MODEL_NAME}_{TILE_SIZE}"
else:
    assert False

##
if is_expression_vae:
    from models.expression_vae import VAE
elif is_image_expression_conv_vae:
    from models.image_expression_conv_vae import VAE
else:
    assert False

##
# if e_():
from utils import file_path

pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
study = optuna.load_study(
    study_name=MODEL_FULLNAME,
    storage="sqlite:///" + file_path(f"optuna_{MODEL_FULLNAME}.sqlite"),
)
print("best trial:")
print(study.best_trial)
print(study.best_trial.user_attrs["version"])

version = study.best_trial.user_attrs["version"]

##
# if e_():
MODEL_CHECKPOINT = file_path(
    f"checkpoints/{MODEL_FULLNAME}/version_{version}/checkpoints/last.ckpt"
)
model = VAE.load_from_checkpoint(MODEL_CHECKPOINT)

##
sample = visium_endrometrium_samples[0]
images, expressions, _ = get_data_per_sample(
    visium_endrometrium_samples[0], tile_dim="large"
)
if is_expression_vae:
    a, b, mu, std, z = model(torch.tensor(expressions))
elif is_image_expression_conv_vae:
    im = torch.tensor(images).permute(0, 3, 1, 2)
    a, b, mu, std, z = model(torch.tensor(expressions), im)

    def get_piece_of_model(model, name):
        piece = [c[1] for c in model.named_children() if c[0] == name]
        assert len(piece) == 1
        return piece[0]
    cnn = get_piece_of_model(model, 'cond_cnn')
    cond_fc = get_piece_of_model(model, 'cond_fc')

    cnn = cnn.cuda()
    im = im.cuda()
    image_activations = cnn(im)
    print(image_activations.shape)

    cond_fc = cond_fc.cuda()
    image_features = cond_fc(torch.flatten(image_activations, start_dim=1))
    print(image_features.shape)

    image_features = ad.AnnData(image_features.detach().cpu().numpy())
else:
    assert False
mu, std, z = [ad.AnnData(t.detach().numpy()) for t in [mu, std, z]]
reconstructed = ad.AnnData(model.expected_value(a, b).detach().numpy())
##
scanpy_compute(z)
##
if is_image_expression_conv_vae:
    scanpy_compute(image_features)
##
if is_image_expression_conv_vae:
    sc.pl.umap(image_features, color='louvain', title='image features louvain')
##
s = get_smu_file(sample, read_only=True)
e = s["visium"]["processed"].clone()
ss = z.obs.louvain.copy()
ss.index = e.masks.obs.index
e.masks.obs["latent_louvain"] = ss

if is_image_expression_conv_vae:
    ss = image_features.obs['louvain']
    ss.index = e.masks.obs.index
    e.masks.obs['image_features_louvain'] = ss
    plt.figure()
    ax = plt.gca()
    e.masks.plot('image_features_louvain', ax=ax)
    ax.set(title='image features louvain')
    plt.show()
##
_, axes = plt.subplots(1, 2, figsize=(10, 5))

s["visium"]["image"].plot(ax=axes[0])
e.masks.plot("latent_louvain", ax=axes[0])
axes[0].set(title="latent louvain")

s["visium"]["image"].plot(ax=axes[1])
e.masks.plot("slide_leiden", ax=axes[1])
axes[1].set(title="slide leiden")

plt.show()
##
feature = 'mean_spot_factors'
f = s['visium'][feature]
axes = plt.subplots(4, 5, figsize=(20, 15))[1].flatten()
for i, ax in enumerate(tqdm(axes)):
    if i >= len(f.var):
        ax.set_axis_off()
        continue
    else:
        v = f.var['channel_name'].tolist()[i]
    f.plot(v, ax=ax, show_title=False)
    ax.set(title=v.replace(feature, ''))
plt.tight_layout()
plt.show()
##
x = s['visium'][feature].X[...]
axes = plt.subplots(4, 5, figsize=(20, 15))[1].flatten()
for i, ax in enumerate(tqdm(axes)):
    if i >= len(f.var):
        ax.set_axis_off()
        continue
    else:
        v = s['visium'][feature].var['channel_name'].tolist()[i]
    ax.hist(x[:, i].flatten())
    ax.set(title=v.replace(feature, ''))
plt.tight_layout()
plt.show()
##
dominant_factors = np.argmax(x / np.std(x, axis=0), axis=1)
print(np.unique(dominant_factors, return_counts=True))
ss = pd.Series(dominant_factors, index=e.obs.index, dtype='category')
e.obs['dominant_factor'] = ss
e.masks.plot('dominant_factor')

##
from analyses.analisys_utils import compare_clusters

##
s.backing.close()
##

ss = e.masks.obs["slide_leiden"]
ss.index = z.obs.index
z.obs["slide_leiden"] = ss

compare_clusters(
    z,
    z,
    description="slide leiden vs latent louvain",
    key0="slide_leiden",
    key1="louvain",
)
##
z.obs['dominant_factor'] = pd.Series(dominant_factors, index=z.obs.index, dtype='category')
compare_clusters(
    z,
    z,
    description="dominant factor vs slide leiden",
    key0="dominant_factor",
    key1="slide_leiden",
)
##
compare_clusters(
    z,
    z,
    description="dominant factor vs latent louvain",
    key0="dominant_factor",
    key1="louvain",
)
##
if is_image_expression_conv_vae:
    compare_clusters(
        z,
        image_features,
        description="dominant factor vs image features",
        key0="dominant_factor",
        key1="louvain",
    )
##
def visualize_images_for_cluster(
    cluster_labels, selected_cluster, cluster_labels_name: str, images
):
    indices = cluster_labels == selected_cluster
    to_show = images[indices]

    ii = np.random.choice(len(to_show), 25, replace=False)

    axes = plt.subplots(5, 5, figsize=(10, 10))[1].flatten()
    for i, ax in enumerate(tqdm(axes)):
        ax.imshow(images[ii[i]])
    plt.suptitle(f"showing cluster {selected_cluster} from {cluster_labels_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


clusters = e.obs["latent_louvain"].value_counts().index.tolist()
for c in clusters[:5]:
    visualize_images_for_cluster(
        cluster_labels=e.obs["latent_louvain"],
        selected_cluster=c,
        cluster_labels_name="latent_louvain",
        images=images,
    )
##
if is_image_expression_conv_vae:
    clusters = image_features.obs["louvain"].value_counts().index.tolist()
    for c in clusters[:5]:
        visualize_images_for_cluster(
            cluster_labels=image_features.obs["louvain"],
            selected_cluster=c,
            cluster_labels_name="latent image features louvain",
            images=images,
        )
##
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def scatter_tiles(z: ad.AnnData, images, inset_location = None):
    random_indices = np.random.permutation(len(z))
    fig, ax = plt.subplots(figsize=(24, 24))
    u = z.obsm["X_umap"]
    for j, i in enumerate(tqdm(random_indices[:700], desc="scatterplot with images")):
        image = images[i]
        im = OffsetImage(image, zoom=0.2)
        ab = AnnotationBbox(im, u[j], xycoords="data", frameon=False)
        ax.add_artist(ab)
    ax.set(xlim=(min(u[:, 0]), max(u[:, 0])), ylim=(min(u[:, 1]), max(u[:, 1])))

    if inset_location is None:
        inset_location = [0.15, 0.02, 0.18, 0.18]
    ins = ax.inset_axes(inset_location)
    sc.pl.umap(z, ax=ins, color='louvain')
    plt.tight_layout()
    plt.show()

##
scatter_tiles(z, images)
if is_image_expression_conv_vae:
    scatter_tiles(image_features, images, inset_location=[0.60, 0.02, 0.18, 0.18])

##
