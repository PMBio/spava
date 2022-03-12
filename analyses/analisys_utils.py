import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

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

