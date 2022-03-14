import anndata as ad
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


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


def compute_knn(b: ad.AnnData):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(b.X)
    distances, indices = nbrs.kneighbors(b.X)
    b.obsm["nearest_neighbors"] = indices
    print("done")


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
    plt.title(
        f"{description}\nadjusted rand score: {adjusted_rand_score(c0, c1):.02f}",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


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
