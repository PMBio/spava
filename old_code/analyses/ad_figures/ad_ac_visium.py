##
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import squidpy as sq

import numpy as np
import pandas as pd

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# load the pre-processed dataset
img = sq.datasets.visium_hne_image()
adata = sq.datasets.visium_hne_adata()

##
plt.figure(figsize=(10, 10))
ax = plt.gca()
sc.pl.spatial(adata, ax=ax)
plt.show()
