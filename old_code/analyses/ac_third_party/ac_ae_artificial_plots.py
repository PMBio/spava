##
import matplotlib.pyplot as plt
from old_code.models import VAE
from analyses.ac_third_party.ac_ad_artificial import FakeRGBCells
from tqdm import tqdm
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd

##
f_original = (
    f"/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/ai_fake/version_30"
    "/checkpoints/last.ckpt"
)
model = VAE.load_from_checkpoint(f_original)
model.cuda()

ds = FakeRGBCells("validation")

latent = []
for i in tqdm(range(len(ds))):
    x, mask, is_perturbed = ds[i]
    x = x.to(model.device).unsqueeze(0)
    mask = mask.to(model.device).unsqueeze(0)
    alpha, beta, mu, std, z = model(x, mask)
    # dist = model.get_dist(alpha, beta)
    # mean = dist.mean
    latent.append(mu.detach().cpu().numpy())
z = np.concatenate(latent, axis=0)

##
print(z.shape)
a = ad.AnnData(z)
sc.tl.pca(a)
sc.pl.pca(a)

##
sc.pp.neighbors(a)
sc.tl.umap(a)
sc.pl.umap(a)
##

plt.style.use("dark_background")
len(ds.filter)
s = pd.Series(ds.filter)
s.index = a.obs.index
a.obs["filter"] = s
sc.pl.umap(a, color=["filter"])
plt.style.use("default")
a.obs["filter"]
ds.filter


##
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

for c in tqdm([0], desc="channels", position=0, leave=False):
    fig, ax = plt.subplots(figsize=(24, 14))
    ax.set(title=f"ch {c}")
    u = a.obsm["X_umap"]
    for i in range(len(ds)):
        ome, _, _ = ds[i]
        ome = ome[c, :, :].numpy()
        # mask = torch.squeeze(mask, 0).numpy()
        im = OffsetImage(ome, zoom=0.7)
        ab = AnnotationBbox(im, u[i], xycoords="data", frameon=False)
        ax.add_artist(ab)
    ax.set(xlim=(min(u[:, 0]), max(u[:, 0])), ylim=(min(u[:, 1]), max(u[:, 1])))
    plt.tight_layout()
    plt.show()
##
