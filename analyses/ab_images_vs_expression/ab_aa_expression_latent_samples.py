##
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import torch
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import matplotlib.cm

from data2 import PerturbedRGBCells, PerturbedCellDataset, IndexInfo
import matplotlib.pyplot as plt
from models.ah_expression_vaes_lightning import VAE as ExpressionVAE, get_loaders
import scanpy as sc
import anndata as ad
import seaborn as sns
import pandas as pd
import optuna

##
SPLIT = "train"
ds = PerturbedRGBCells(split=SPLIT)
cells_ds = PerturbedCellDataset(split=SPLIT)

PERTURB = False
if PERTURB:
    ds.perturb()
    cells_ds.perturb()
assert np.all(ds.corrupted_entries.numpy() == cells_ds.corrupted_entries.numpy())

##
# train the model on dilated masks using the hyperparameters from the best model for original expression
from models.ah_expression_vaes_lightning import objective, ppp
from data2 import file_path
pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
study_name = "no-name-fbdac942-b370-43af-a619-621755ee9d1f"
ppp.PERTURB_MASKS = True
study = optuna.load_study(study_name=study_name, storage="sqlite:///" + file_path("optuna_ah.sqlite"))
objective(study.best_trial)

##
ii = IndexInfo(SPLIT)
n = ii.filtered_ends[-1]
random_indices = np.random.choice(n, 10000, replace=False)


def precompute(data_loaders, expression_model_checkpoint, random_indices):
    loader = data_loaders[["train", "validation", "test"].index(SPLIT)]
    expression_vae = ExpressionVAE.load_from_checkpoint(expression_model_checkpoint)
    print("merging expressions and computing embeddings... ", end="")
    all_mu = []
    all_expression = []
    for data in tqdm(loader, desc="embedding expression"):
        expression, _, is_perturbed = data
        a, b, mu, std, z = expression_vae(expression)
        all_mu.append(mu)
        all_expression.append(expression)
    mus = torch.cat(all_mu, dim=0)
    expressions = torch.cat(all_expression, dim=0)
    print("done")

    a0 = ad.AnnData(mus.detach().numpy())
    sc.tl.pca(a0)
    sc.pl.pca(a0)
    a1 = ad.AnnData(expressions.numpy())

    b0 = a0[random_indices]
    b1 = a1[random_indices]
    ##
    # a, b = a0, b0
    # for a, b in tqdm(zip([a0, a1], [b0, b1]), desc="AnnData objects", total=2):
    for b in tqdm([b0, b1], desc="AnnData objects", total=2):
        print("computing umap... ", end="")
        sc.pp.neighbors(b)
        sc.tl.umap(b)
        sc.tl.louvain(b)
        print("done")

        print("computing nearest neighbors on subsetted data... ", end="")
        nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(b.X)
        distances, indices = nbrs.kneighbors(b.X)
        b.obsm["nearest_neighbors"] = indices
        print("done")

        # print("computing nearest neighbors on the full data... ", end="")
        # nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(a.X)
        # distances, indices = nbrs.kneighbors(a.X)
        # a.obsm["nearest_neighbors"] = indices
        # print("done")
    # return a0, a1, b0, b1
    return b0, b1


##
# best expression model
b0, b1 = precompute(
    data_loaders=get_loaders(perturb=PERTURB),
    expression_model_checkpoint="/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints"
    "/expression_vae/version_57/checkpoints/last.ckpt",
    random_indices=random_indices,
)

##
# best expression model trainined on dilated masks
b0m, b1m = precompute(
    get_loaders(perturb=PERTURB, perturb_masks=True),
    expression_model_checkpoint="/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints"
    "/expression_vae/version_108/checkpoints/last.ckpt",
    random_indices=random_indices,
)


##
def louvain_plot(an: ad.AnnData, title: str):
    plt.figure()
    l = an.obs["louvain"].tolist()
    colors = list(map(int, l))
    plt.scatter(
        an.obsm["X_umap"][:, 0],
        an.obsm["X_umap"][:, 1],
        s=1,
        c=colors,
        cmap=matplotlib.cm.tab20,
    )
    plt.title(title)
    plt.show()


##
louvain_plot(b0, title="expression latent space")
louvain_plot(b1, title="expression space")
louvain_plot(b0m, title="expression latent space (dilated masks)")
louvain_plot(b1m, title="expression space (dilated masks)")


##
def compare_clusters(an0: ad.AnnData, an1: ad.AnnData, description: str):
    c0 = np.array(list(map(int, an0.obs["louvain"].tolist())))
    c1 = np.array(list(map(int, an1.obs["louvain"].tolist())))
    m = np.zeros((c0.max() + 1, c1.max() + 1))
    print(m.shape)
    import itertools

    for x0, x1 in itertools.product(range(m.shape[0]), range(m.shape[1])):
        z0 = np.where(c0 == x0)[0]
        z1 = c1[z0]
        z2 = z1[z1 == x1]
        m[x0, x1] = len(z2) / len(z1)
    df = pd.DataFrame(m)
    a = sns.clustermap(df)
    plt.close()

    mm = a.data2d.to_numpy()
    m0 = np.argmax(mm, axis=0)
    m1 = sorted(zip(range(len(m0)), m0), key=lambda x: x[1])
    m2, m3 = zip(*m1)
    mm0 = mm[:, m2]
    plt.figure(figsize=(9, 9))
    plt.imshow(mm0)
    plt.colorbar()
    plt.title(f"{description}: adjusted rand score: {adjusted_rand_score(c0, c1)}")
    plt.show()


##
compare_clusters(b1, b1m, description='"expression" vs "expression (dilated masks)"')
compare_clusters(b0, b0m, description='"latent" vs "latent (dilated masks)"')
compare_clusters(b1, b0, description='"expression" vs "latent"')
compare_clusters(
    b1m, b0m, description='"expression (dilated masks)" vs "latent (dilated masks)"'
)


##
def nearest_neighbors(nn_from: ad.AnnData, plot_onto: ad.AnnData, title: str):
    some_cells = list(range(5))
    plt.style.use("dark_background")
    # plot the nearest neighbors on the umaps
    axes = plt.subplots(1, len(some_cells), figsize=(15, 5))[1].flatten()
    k = 0
    for cell in tqdm(some_cells):
        indices = nn_from.obsm["nearest_neighbors"]
        nn = indices[cell]
        ax = axes[k]
        k += 1
        ax.scatter(
            plot_onto.obsm["X_umap"][:, 0],
            plot_onto.obsm["X_umap"][:, 1],
            s=1,
            color=(0.3, 0.3, 0.3),
        )
        ax.scatter(
            plot_onto.obsm["X_umap"][nn, 0],
            plot_onto.obsm["X_umap"][nn, 1],
            s=1,
            color="w",
        )
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.style.use("default")

##
nearest_neighbors(
    nn_from=b1,
    plot_onto=b1,
    title='nn from "expression" plotted plotted onto "expression"',
)
nearest_neighbors(
    nn_from=b1,
    plot_onto=b1m,
    title='nn from "expression" plotted plotted onto "expression (dilated masks)"',
)
nearest_neighbors(
    nn_from=b1,
    plot_onto=b0,
    title='nn from "expression" plotted plotted onto "latent space"',
)
nearest_neighbors(
    nn_from=b1,
    plot_onto=b0m,
    title='nn from "expression" plotted plotted onto "latent space (dilated masks)"',
)
##
if False:
    # plot barplots of expression for nearest neighbors, of subsampled cells
    some_cells = range(5)
    rows = len(some_cells)
    cols = len(indices[42])
    axes = plt.subplots(rows, cols, figsize=(30, 20))[1].flatten()
    k = 0
    for cell in tqdm(some_cells):
        for i in indices[cell]:
            ax = axes[k]
            e = c1[i]
            ax.bar(np.arange(len(e)), e)
            ax.set(title=f"cell {i}, nn of {cell}")
            k += 1
    plt.tight_layout()
    plt.show()
