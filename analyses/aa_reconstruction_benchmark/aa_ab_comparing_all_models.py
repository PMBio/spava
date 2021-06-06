##
import math

import pytorch_lightning as pl

pl.seed_everything(1234)
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

##
from models.ah_expression_vaes_lightning import PerturbedCellDataset

ds0 = PerturbedCellDataset('train')
ds0.perturb()
ds1 = PerturbedCellDataset('train')
ds1.perturb()
assert torch.all(ds0.corrupted_entries == ds1.corrupted_entries)

##
models = {
    'nb': '/data/l989o/data/basel_zurich/spatial_uzh_processed/a/checkpoints/expression_vae'
          '/version_45/checkpoints/last.ckpt',
    'gaussian': '/data/l989o/data/basel_zurich/spatial_uzh_processed/a/checkpoints/expression_vae'
                '/version_46/checkpoints/last.ckpt',
    'zip': '/data/l989o/data/basel_zurich/spatial_uzh_processed/a/checkpoints/expression_vae'
           '/version_49/checkpoints/last.ckpt',
    'zin': '/data/l989o/data/basel_zurich/spatial_uzh_processed/a/checkpoints/expression_vae'
           '/version_51/checkpoints/last.ckpt',
}
better_title = {
    'gaussian': 'Gaussian',
    'nb': 'Gamma-Poisson',
    'zip': 'Zero-inflated Poisson',
    'zin': 'Zero-inflated Gaussian'
}
##
from models.ah_expression_vaes_lightning import VAE, set_ppp_from_loaded_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import time

fig, axes = plt.subplots(math.ceil(len(models) / 2), 2, figsize=(10, 10))
axes = axes.flatten(axes)
for i, model_title in enumerate(tqdm(models, desc='iterating through models ')):
    checkpoint = models[model_title]

    model = VAE.load_from_checkpoint(checkpoint)
    set_ppp_from_loaded_model(model)

    x = ds0.original_merged
    x_zero = ds0.merged
    x_pred = model.forward(x)[0]
    x_zero_pred = model.forward(x_zero)[0]

    x_pred_mean = model.expected_value(x_pred)
    x_zero_pred_mean = model.expected_value(x_zero_pred)

    a = ds0.corrupted_entries
    score = torch.median(torch.abs(x[a] - x_zero_pred_mean[a]))
    print(score)

    v = x[a].detach().numpy()
    w = x_zero_pred_mean[a].detach().numpy()

    imputed = w
    original = v

    cutoff = 2
    mask = imputed < cutoff
    imputed = imputed[mask]
    original = original[mask]

    mask = original < cutoff
    imputed = imputed[mask]
    original = original[mask]

    l = np.minimum(imputed.shape[0], original.shape[0])

    assert len(imputed) == len(original)
    imputed = imputed[:l]
    original = original[:l]

    a, _, _, _ = np.linalg.lstsq(original[:, np.newaxis], imputed)
    l = np.linspace(0, cutoff)

    axes[i].scatter(v, w, s=1, alpha=0.05)
    axes[i].set_title(better_title[model_title])
    axes[i].set_xlabel('original')
    axes[i].set_ylabel('imputed')
    axes[i].set(xlim=(0, 2), ylim=(0, 2))
    axes[i].plot(l, a * l, color='black')

    A = np.vstack([original, np.ones(len(original))]).T
    aa, _, _, _ = np.linalg.lstsq(A, imputed)
    axes[i].plot(l, aa[0] * l + aa[1], color='red')

    axes[i].plot(l, l, color='black', linestyle=":")
plt.show()

##
print('done')
