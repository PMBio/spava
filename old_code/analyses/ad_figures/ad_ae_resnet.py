# %%

import os
import sys

os.chdir("/data/l989o/deployed/a")
if "/data/l989o/projects/a" in sys.path:
    sys.path.remove("/data/l989o/projects/a")

# %%

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np

# import pl_bolts.utils
import torch
from torch.utils.data import DataLoader
from old_code.models import VAE, RGBCells, COOL_CHANNELS
import matplotlib.pyplot as plt
from torchvision import transforms
import PIL

# %%

# model_path = '/data/l989o/spatial_uzh/data/spatial_uzh_processed/a/checkpoints/resnet_vae/version_115/checkpoints'
model_path = "/data/l989o/data/basel_zurich/spatial_uzh_processed/a/checkpoints/resnet_vae/version_19/checkpoints"
l = os.listdir(model_path)
assert len(l) == 1
assert l[0].endswith(".ckpt")
l

# %%

checkpoint = os.path.join(model_path, l[0])
model = VAE.load_from_checkpoint(checkpoint)
model = model.cuda()
val_ds = RGBCells("validation")
val_loader = DataLoader(
    val_ds, batch_size=16, num_workers=8, pin_memory=True, shuffle=True
)

# %%

quantiles_for_normalization = np.array(
    [
        4.0549,
        1.8684,
        1.3117,
        3.8141,
        2.6172,
        3.1571,
        1.4984,
        1.8866,
        1.2621,
        3.7035,
        3.6496,
        1.8566,
        2.5784,
        0.9939,
        1.4314,
        2.1803,
        1.8672,
        1.6674,
        2.3555,
        0.8917,
        5.1779,
        1.8002,
        1.4042,
        2.3873,
        1.0509,
        1.0892,
        2.2708,
        3.4417,
        1.8348,
        1.8449,
        2.8699,
        2.2071,
        1.0464,
        2.5855,
        2.0384,
        4.8609,
        2.0277,
        3.3281,
        3.9273,
    ]
)[COOL_CHANNELS]

# %%

from old_code.models import get_image

img, all_original_c, all_reconstructed_c, all_masks = get_image(
    val_loader, model, return_cells=True
)

# %%

print(len(all_original_c))
print(len(all_reconstructed_c))
print(len(all_masks))
print(len(all_original_c[0]))

# %%

assert len(COOL_CHANNELS) == len(all_original_c)
all_p = torch.sigmoid(model.negative_binomial_p_logit)
for channel in range(len(COOL_CHANNELS)):
    p = all_p[channel]
    for i, mask in enumerate(all_masks):
        original = all_original_c[channel][i]
        reconstructed = all_reconstructed_c[channel][i]
        # gray scale image with repeated channels
        assert torch.all(original[0] == original[1])
        assert torch.all(original[0] == original[2])
        assert torch.all(reconstructed[0] == reconstructed[1])
        assert torch.all(reconstructed[0] == reconstructed[2])
        original = original[0]
        reconstructed = reconstructed[0]
        plt.figure()
        plt.imshow(original, cmap="gray")
        plt.show()

        plt.figure()
        plt.imshow(reconstructed, cmap="gray")
        plt.show()

        if i >= 1:
            break
    else:
        continue
    break

# %%

n = 15

mask_color = torch.tensor([x / 255 for x in [254, 112, 31]]).float()
new_size = (128, 128)
upscale = transforms.Resize(new_size, interpolation=PIL.Image.NEAREST)
n = 15
with torch.no_grad():
    batch = val_loader.__iter__().__next__()
    omes = batch[0]
    masks = batch[1]
    assert len(omes.shape) == 4
    assert len(omes) >= n
    data = omes[:n].to(model.device)
    masks_data = masks[:n].to(model.device)
    pred = model.forward(data, masks_data)[0]
n_channels = data.shape[1]

# %%

x = np.zeros((3, 3, 3))
y = np.ones((3, 3), dtype=np.bool)
y[1, 1] = 0
print(y)
x[y, :] = 2
x

# %%

x = torch.zeros(3, 3, 3)
y = torch.ones(3, 3, dtype=torch.bool)
y[1, 1] = 0
print(y)
x[y, :] = 2
x

# %%

for i in range(5):
    c = 0

    original = data[i].cpu().permute(1, 2, 0) * quantiles_for_normalization
    r_hat = pred[i].cpu().permute(1, 2, 0)
    p = torch.sigmoid(model.negative_binomial_p_logit).cpu().detach()
    mean = model.negative_binomial_mean(r=r_hat, p=p)
    reconstructed = mean * quantiles_for_normalization

    a_original = original.amin(dim=(0, 1))
    b_original = original.amax(dim=(0, 1))
    m = masks_data[i].cpu().bool()
    mm = torch.squeeze(m, 0)
    mm_not = torch.logical_not(mm)
    reconstructed_flattened = torch.reshape(
        reconstructed, (-1, reconstructed.shape[-1])
    )
    mask_flattened = mm.flatten()
    if mask_flattened.sum() > 0:
        a_reconstructed = reconstructed_flattened[mask_flattened, :].amin(dim=0)
        b_reconstructed = reconstructed_flattened[mask_flattened, :].amax(dim=0)
        a = torch.min(a_original, a_reconstructed)
        b = torch.max(b_original, b_reconstructed)

        original = ((original - a) / (b - a)).float()
        reconstructed = ((reconstructed - a) / (b - a)).float()

        axes = plt.subplots(2, 2, figsize=(10, 9))[1].flatten()
        axes[0].imshow(mm, cmap="gray")
        axes[0].set(title="mask")

        original[mm_not, :] = 0
        axes[1].imshow(original[:, :, c], cmap="gray")
        axes[1].set(title="original")
        max_original = original.max()

        reconstructed[mm_not, :] = 0
        axes[2].imshow(reconstructed[:, :, c], cmap="gray", vmax=max_original)
        axes[2].set(title="distribution")

        x0 = r_hat[:, :, c].numpy()
        x1 = p[c].repeat(1024).reshape(32, 32).numpy()
        assert x0.shape == x1.shape
        assert x0.shape == (32, 32)
        from scipy.stats import nbinom

        # THERE IS A BUG HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        # THERE IS A BUG HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        # THERE IS A BUG HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        # THERE IS A BUG HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        # THERE IS A BUG HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        # THERE IS A BUG HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE the range of x is wrong (set vmax and see)
        x = nbinom.rvs(x0, x1, size=(32, 32))  # * quantiles_for_normalization[c]
        x[mm_not.numpy()] = 0
        axes[3].imshow(x, cmap="gray")  # , vmax=max_original)
        axes[3].set(title="simulated")
        plt.show()
    else:
        print("empty mask")
