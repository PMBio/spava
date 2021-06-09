##
import numpy as np
import time
import torch
from tqdm import tqdm

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
    'resnet_vae_perturbed': '/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_12'
                            '/checkpoints/last.ckpt',
    'resnet_vae_perturbed_long': '/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_14'
                                 '/checkpoints/last.ckpt'
}

rgb_ds = ds.rgb_cells
from models.ag_conv_vae_lightning import VAE as ResNetVAE

resnet_vae = ResNetVAE.load_from_checkpoint(models['resnet_vae'])
loader = DataLoader(rgb_ds, batch_size=16, num_workers=8, pin_memory=True)
data = loader.__iter__().__next__()

resnet_vae_perturbed = ResNetVAE.load_from_checkpoint(models['resnet_vae_perturbed'])
loader_perturbed = DataLoader(ds, batch_size=16, num_workers=16, pin_memory=True)
data_perturbed = loader_perturbed.__iter__().__next__()
##
renset_vae = resnet_vae.cuda()
data = [d.to(resnet_vae.device) for d in data]

resnet_vae_perturbed = resnet_vae_perturbed.cuda()
data_perturbed = [d.to(resnet_vae_perturbed.device) for d in data_perturbed]

##
start = time.time()
with torch.no_grad():
    z = [zz.cpu() for zz in resnet_vae(*data)]
    z_perturbed = [zz.cpu() for zz in resnet_vae_perturbed(*(data_perturbed[:2]))]
print(f'forwarning the data to the resnets: {time.time() - start}')

##
torch.cuda.empty_cache()
##
from models.ag_conv_vae_lightning import quantiles_for_normalization, get_image
from data2 import CHANNEL_NAMES

image = get_image(loader, resnet_vae)
np_image = image.permute(1, 2, 0).numpy()
# from PIL import Image
# image = Image.fromarray(np.uint8(np_image * 255))
# image.show()
##
assert (resnet_vae_perturbed.encoder.mask_conv1.weight == resnet_vae_perturbed.decoder.mask_conv1.weight).prod()
##
alpha0, beta0, mu0, std0, z0 = z
alpha1, beta1, mu1, std1, z1 = z_perturbed

mask0 = data[1]
mask1 = data_perturbed[1]
assert (mask0 == mask1).all()

ome_index = 8
channel = 0

mask = mask0.bool()[ome_index].squeeze(dim=0)
inverted_mask = mask.logical_not()

alpha0 = alpha0[ome_index]
beta0 = beta0[ome_index]
alpha1 = alpha1[ome_index]
beta1 = beta1[ome_index]


def f(values):
    values_masked = values.clone()
    for c in range(len(values)):
        channel = values[c]
        v = channel[mask].min()
        values_masked[c, inverted_mask] = v
    return values_masked


alpha0_m = f(alpha0)
beta0_m = f(beta0)
alpha1_m = f(alpha1)
beta1_m = f(beta1)
# ##
if False:
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.title('alpha')
    plt.imshow(alpha0_m[channel, :, :])
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('beta')
    plt.imshow(beta0_m[channel, :, :])
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('alpha (perturbed model)')
    plt.imshow(alpha1_m[channel, :, :])
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('beta (perturbed model)')
    plt.imshow(beta1_m[channel, :, :])
    plt.colorbar()

    plt.tight_layout()
    plt.show()

dist0 = resnet_vae.get_dist(alpha0_m, beta0_m)
dist1 = resnet_vae_perturbed.get_dist(alpha1_m, beta1_m)

# plotting the original data against the mean of the standard and the perturbed model
mean0 = dist0.mean
mean1 = dist1.mean
mask0 = data[1]
mask = mask0.bool()[ome_index].squeeze(dim=0)
inverted_mask = mask.logical_not()

mean0[:, inverted_mask] = 0.
mean1[:, inverted_mask] = 0.
n = 39
# n = 5
axes = plt.subplots(n, 3, figsize=(10, 90))[1].flatten()
for i, (original, m0, m1) in enumerate(zip(data[0][ome_index].cpu(), mean0, mean1)):
    if i >= n:
        break
    ax = axes[3 * i]
    ax.imshow(original, cmap='gray')
    ax = axes[3 * i + 1]
    ax.imshow(m0, cmap='gray')
    ax.set(title='non-perturbed')
    ax = axes[3 * i + 2]
    ax.imshow(m1, cmap='gray')
    ax.set(title='perturbed')
plt.tight_layout()
plt.show()

# plotting a sample from the predicted distributions
s0 = dist0.sample()
s1 = dist1.sample()

s0[:, inverted_mask] = 0.
s1[:, inverted_mask] = 0.
n = 39
# n = 5
axes = plt.subplots(n, 3, figsize=(10, 90))[1].flatten()
for i, (original, m0, m1) in enumerate(zip(data[0][ome_index].cpu(), s0, s1)):
    if i >= n:
        break
    ax = axes[3 * i]
    ax.imshow(original, cmap='gray')
    ax = axes[3 * i + 1]
    ax.imshow(m0, cmap='gray')
    ax.set(title='non-perturbed')
    ax = axes[3 * i + 2]
    ax.imshow(m1, cmap='gray')
    ax.set(title='perturbed')
plt.tight_layout()
plt.show()
##
# plotting the average of pixels vs the average of the reconstructed, here I am recomputeing the averages so that I
# don't have to preprocess or "inverse-preprocess" things
##
start = time.time()
list_of_z = []
with torch.no_grad():
    for data in tqdm(loader, desc='embedding the whole validation set'):
        data = [d.to(resnet_vae.device) for d in data]
        z = [zz.cpu() for zz in resnet_vae(*data)]
        list_of_z.append(z)
print(f'forwarning the data to the resnets: {time.time() - start}')
##
from data2 import file_path

f = file_path('image_features_resnet_vae_gamma_poisson.h5py')
import h5py

if False:
    len(list_of_z)
    len(list_of_z[0])
    alpha, beta, mu, std, z = list(zip(*list_of_z))
    with h5py.File(f, 'w') as f5:
        f5['alpha'] = torch.cat(alpha, dim=0).numpy()
        f5['beta'] = torch.cat(beta, dim=0).numpy()
        f5['mu'] = torch.cat(mu, dim=0).numpy()
        f5['std'] = torch.cat(std, dim=0).numpy()

if True:
    for data in tqdm(loader, desc='putting the masks in the hdf5 file')
    with h5py.File(f, 'a') as f5:
        pass

##
with h5py.File(f, 'r') as f5:
    n_cells = len(f5['alpha'])
    print(n_cells)
    ii = np.random.choice(10000, n_cells)
    lucky_cells