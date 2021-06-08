##
import numpy as np
import time
import torch

from models.ag_conv_vae_lightning import RGBCells
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models.ag_conv_vae_lightning import PerturbedRGBCells
from models.ah_expression_vaes_lightning import PerturbedCellDataset

ds = PerturbedRGBCells(split='validation')
ds.perturb()

cells_ds = PerturbedCellDataset(split='validation')
cells_ds.perturb()

assert torch.all(ds.corrupted_entries == cells_ds.corrupted_entries)
##
print(ds.corrupted_entries.shape)
print(cells_ds.corrupted_entries.shape)
print(ds.corrupted_entries[0, :])
print(cells_ds.corrupted_entries[0, :])
##
i = 10
c = ds.corrupted_entries[i, :]
print(c)
j = torch.nonzero(c).flatten()[0].item()
##
x = ds[i][0][j]
x_original = ds.rgb_cells[i][0][j]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(x)
plt.subplot(1, 2, 2)
plt.imshow(x_original)
plt.show()

##

models = {
    'resnet_vae': '/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_7/checkpoints'
                  '/epoch=3-step=1610.ckpt',
    # 'resnet_vae_perturbed':
}

rgb_ds = ds.rgb_cells
from models.ag_conv_vae_lightning import VAE as ResNetVAE

resnet_vae = ResNetVAE.load_from_checkpoint(models['resnet_vae'])
loader = DataLoader(rgb_ds, batch_size=16, num_workers=8, pin_memory=True)
data = loader.__iter__().__next__()
##
renset_vae = resnet_vae.cuda()
data = [d.to(resnet_vae.device) for d in data]

##
start = time.time()
with torch.no_grad():
    z = [zz.cpu() for zz in resnet_vae(*data)]
print(f'forwarning the data to the resnet: {time.time() - start}')

##
torch.cuda.empty_cache()
##
from models.ag_conv_vae_lightning import quantiles_for_normalization, get_image
from data2 import CHANNEL_NAMES

image = get_image(loader, resnet_vae)
np_image = image.permute(1, 2, 0).numpy()
from PIL import Image
image = Image.fromarray(np.uint8(np_image * 255))
image.show()