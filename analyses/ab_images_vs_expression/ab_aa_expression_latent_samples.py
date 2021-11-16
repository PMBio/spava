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
from utils import reproducible_random_choice

m = __name__ == '__main__'

STUDY_DILATED_MASK = False
STUDY_PERTURBED_PIXELS = False
STUDY_STABILITY = False

##
if m:
    SPLIT = "train"
    ds = PerturbedRGBCells(split=SPLIT)
    cells_ds = PerturbedCellDataset(split=SPLIT)

    PERTURB = False
    if PERTURB:
        ds.perturb()
        cells_ds.perturb()
    assert np.all(ds.corrupted_entries.numpy() == cells_ds.corrupted_entries.numpy())

##
if m and STUDY_DILATED_MASK and False:
    # train the model on dilated masks using the hyperparameters from the best model for original expression
    from models.ah_expression_vaes_lightning import objective, ppp
    from data2 import file_path

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = "no-name-fbdac942-b370-43af-a619-621755ee9d1f"
    ppp.PERTURB_MASKS = True
    study = optuna.load_study(
        study_name=study_name, storage="sqlite:///" + file_path("optuna_ah.sqlite")
    )
    objective(study.best_trial)

##
if m:
    ii = IndexInfo(SPLIT)
    n = ii.filtered_ends[-1]
    random_indices = reproducible_random_choice(n, 10000)
##

def compute_knn(b: ad.AnnData):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(b.X)
    distances, indices = nbrs.kneighbors(b.X)
    b.obsm["nearest_neighbors"] = indices
    print("done")

def precompute(data_loaders, expression_model_checkpoint, random_indices, split):
    loader = data_loaders[["train", "validation"].index(split)]
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
    # a, b = a0, b0
    # for a, b in tqdm(zip([a0, a1], [b0, b1]), desc="AnnData objects", total=2):
    for b in tqdm([b0, b1], desc="AnnData objects", total=2):
        print("computing umap... ", end="")
        sc.pp.neighbors(b)
        sc.tl.umap(b)
        sc.tl.louvain(b)
        print("done")

        print("computing nearest neighbors on subsetted data... ", end="")
        compute_knn(b)

        # print("computing nearest neighbors on the full data... ", end="")
        # nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(a.X)
        # distances, indices = nbrs.kneighbors(a.X)
        # a.obsm["nearest_neighbors"] = indices
        # print("done")
    # return a0, a1, b0, b1
    return b0, b1


##
if m:
    # best expression model
    # 57
    b0, b1 = precompute(
        data_loaders=get_loaders(perturb=PERTURB),
        expression_model_checkpoint="/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints"
        "/expression_vae/version_145/checkpoints/last.ckpt",
        random_indices=random_indices,
        split=SPLIT
    )

##
if m and STUDY_DILATED_MASK:
    # best expression model trainined on dilated masks
    b0m, b1m = precompute(
        get_loaders(perturb=PERTURB, perturb_masks=True),
        expression_model_checkpoint="/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints"
        "/expression_vae/version_108/checkpoints/last.ckpt",
        random_indices=random_indices,
        split=SPLIT
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
if m:
    louvain_plot(b0, title="expression latent space")
    louvain_plot(b1, title="expression space")
    if STUDY_DILATED_MASK:
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
    dpi = 100
    plt.figure(figsize=(400 / dpi, 400 / dpi), dpi=dpi)
    plt.imshow(mm0)
    plt.colorbar()
    plt.title(f"{description}\nadjusted rand score: {adjusted_rand_score(c0, c1):.02f}", y=1.02)
    plt.tight_layout()
    plt.show()


##
if m:
    compare_clusters(b1, b0, description='"expression" vs "latent"')
    from data2 import file_path
    import pickle
    pickle.dump({'input': b1, 'latent': b0}, open(file_path('latent_anndata_from_ah_model.pickle'), 'wb'))
    if STUDY_DILATED_MASK:
        compare_clusters(b1, b1m, description='"expression" vs "expression (dilated masks)"')
        compare_clusters(b0, b0m, description='"latent" vs "latent (dilated masks)"')
        compare_clusters(
            b1m, b0m, description='"expression (dilated masks)" vs "latent (dilated masks)"'
        )


##
def compute_knn_purity(ad0: ad.AnnData, ad1: ad.AnnData) -> float:
    nn0 = ad0.obsm["nearest_neighbors"]
    nn1 = ad1.obsm["nearest_neighbors"]
    assert nn0.shape[1] == nn1.shape[1]
    k = nn0.shape[1]
    # let's use a list in the case in which we are interested in a histogram
    scores = []
    for x0, x1 in zip(nn0, nn1):
        common = np.intersect1d(x0, x1, assume_unique=True)
        scores.append(len(common) / k)
    m = np.mean(scores)
    return m.item()


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
    p = compute_knn_purity(nn_from, plot_onto)
    plt.suptitle(f"knn purity: {p:0.2f}, {title}")
    plt.tight_layout()
    plt.show()
    plt.style.use("default")


##
if m:
    nearest_neighbors(
        nn_from=b1,
        plot_onto=b1,
        title='nn from "expression" plotted plotted onto "expression"',
    )
    nearest_neighbors(
        nn_from=b1,
        plot_onto=b0,
        title='nn from "expression" plotted plotted onto "latent space"',
    )
    if STUDY_DILATED_MASK:
        nearest_neighbors(
            nn_from=b1,
            plot_onto=b1m,
            title='nn from "expression" plotted plotted onto "expression (dilated masks)"',
        )
        nearest_neighbors(
            nn_from=b1,
            plot_onto=b0m,
            title='nn from "expression" plotted plotted onto "latent space (dilated masks)"',
        )
##
if m and STUDY_PERTURBED_PIXELS and False:
    from data2 import CellDataset

    r = CellDataset("train").FRACTION_OF_PIXELS_TO_MASK
    print(f"fraction of pixels to mask = {r}")
    # train various models of perturbed pixels using the hyperparameters from the best model for expression
    # training time is approx 5 minutes per model
    for seed in tqdm(range(12), desc=f"perturbed pixels model"):
        print(f"seed = {seed}")
        from models.ah_expression_vaes_lightning import objective, ppp
        from data2 import file_path

        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
        study_name = "no-name-fbdac942-b370-43af-a619-621755ee9d1f"
        ppp.PERTURB_PIXELS = True
        ppp.PERTURB_PIXELS_SEED = seed
        ppp.PERTURB_MASKS = False
        study = optuna.load_study(
            study_name=study_name, storage="sqlite:///" + file_path("optuna_ah.sqlite")
        )
        objective(study.best_trial)
##
if m and STUDY_PERTURBED_PIXELS:
    perturbed_expressions = []
    perturbed_mus = []
    for i in tqdm(range(12), "perturbed model"):
        loader = get_loaders(
            perturb=PERTURB, perturb_pixels=True, perturb_pixels_seed=i, perturb_masks=False
        )[["train", "validation"].index(SPLIT)]
        expression_model_checkpoint = (
            f"/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/expression_vae"
            f"/version_{i + 110}/checkpoints/last.ckpt"
        )
        expression_vae = ExpressionVAE.load_from_checkpoint(expression_model_checkpoint)
        l = []
        for data in tqdm(loader, desc="embedding expression", leave=False):
            expression, _, is_perturbed = data
            l.append(expression)
        expressions = torch.cat(l, dim=0)
        random_expressions = expressions[random_indices]
        # a, b, mu, std, z = expression_vae(expression)
        _, _, random_mu, _, _ = expression_vae(random_expressions)
        perturbed_expressions.append(random_expressions)
        perturbed_mus.append(random_mu)
##
if m and STUDY_PERTURBED_PIXELS:
    a_expr = [ad.AnnData(x.numpy()) for x in perturbed_expressions]
    a_mus = [ad.AnnData(x.detach().numpy()) for x in perturbed_mus]
##
if m and STUDY_PERTURBED_PIXELS:
    for i in tqdm(range(12), desc='computing nearest neighbors'):
        compute_knn(a_expr[i])
        compute_knn(a_mus[i])
##
if m and STUDY_PERTURBED_PIXELS:
    purities_against_expression = []
    purities_against_latent = []
    for i in tqdm(range(12), desc='computing knn purity'):
        p = compute_knn_purity(a_expr[i], b1)
        purities_against_expression.append(p)
        p = compute_knn_purity(a_mus[i], b0m)
        purities_against_latent.append(p)
    p_e = np.array(purities_against_expression)
    p_l = np.array(purities_against_latent)
    print(p_e)
    # gives: [0.77087  0.769275 0.76964  0.770855 0.76818  0.76979  0.77168  0.770635
    #  0.772285 0.7698   0.769245 0.77139 ]
    print(p_l)
    # gives: [0.259405 0.212675 0.22681  0.30572  0.270595 0.19452  0.294845 0.20003
    #  0.24315  0.240395 0.292055 0.21468 ]
##
# assessing the stability of the model (let's retrain a few more times the expression model with the same
# hyperparameters)
if m and STUDY_STABILITY and False:
    for _ in range(3):
        from models.ah_expression_vaes_lightning import objective, ppp
        from data2 import file_path

        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
        study_name = "no-name-fbdac942-b370-43af-a619-621755ee9d1f"
        ppp.PERTURB_PIXELS = False
        ppp.PERTURB_PIXELS_SEED = 42
        ppp.PERTURB_MASKS = False
        study = optuna.load_study(
            study_name=study_name, storage="sqlite:///" + file_path("optuna_ah.sqlite")
        )
        objective(study.best_trial)
##
if m and STUDY_STABILITY:
    b0s = []
    b1s = []
    for i in tqdm(range(3), 'precomputing'):
        bb0, bb1 = precompute(
            data_loaders=get_loaders(perturb=PERTURB),
            expression_model_checkpoint="/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints"
                                        f"/expression_vae/version_{i + 124}/checkpoints/last.ckpt",
            random_indices=random_indices,
            split=SPLIT
        )
        b0s.append(bb0)
        b1s.append(bb1)
##
if m and STUDY_STABILITY:
    for i in range(3):
        louvain_plot(b1s[i], title=f'expression, clone {i}')

    for i in range(3):
        louvain_plot(b0s[i], title=f'latent, clone {i}')

##
if m and STUDY_STABILITY:
    for i in range(3):
        compare_clusters(b1, b1s[i], description=f'"expression" vs "expression, clone {i}"')

    for i in range(3):
        compare_clusters(b0, b0s[i], description=f'"latent" vs "latent, clone {i}"')

    for i in range(3):
        compare_clusters(b1, b0s[i], description=f'"expression" vs "latent, clone {i}"')
##
if m and STUDY_STABILITY:
    for i in range(3):
        nearest_neighbors(
            nn_from=b1,
            plot_onto=b0s[i],
            title=f'nn from "expression" plotted plotted onto "latent space, clone {i}"',
        )
##
if m and STUDY_STABILITY:
    for i in range(3):
        nearest_neighbors(
            nn_from=b0,
            plot_onto=b0s[i],
            title=f'nn from "latent" plotted plotted onto "latent space, clone {i}"',
        )
##
# if False:
#     # plot barplots of expression for nearest neighbors, of subsampled cells
#     some_cells = range(5)
#     rows = len(some_cells)
#     cols = len(indices[42])
#     axes = plt.subplots(rows, cols, figsize=(30, 20))[1].flatten()
#     k = 0
#     for cell in tqdm(some_cells):
#         for i in indices[cell]:
#             ax = axes[k]
#             e = c1[i]
#             ax.bar(np.arange(len(e)), e)
#             ax.set(title=f"cell {i}, nn of {cell}")
#             k += 1
#     plt.tight_layout()
#     plt.show()
