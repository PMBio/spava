##
from skimage import measure
import torch
from tqdm import tqdm

from old_code.data2 import PerturbedRGBCells, PerturbedCellDataset
import matplotlib.pyplot as plt

ds = PerturbedRGBCells(split="validation")

cells_ds = PerturbedCellDataset(split="validation")
if False:
    ds.perturb()
    cells_ds.perturb()

assert torch.all(ds.corrupted_entries == cells_ds.corrupted_entries)
if False:
    print(ds.corrupted_entries.shape)
    print(cells_ds.corrupted_entries.shape)
    print(ds.corrupted_entries[0, :])
    print(cells_ds.corrupted_entries[0, :])
    i = 10
    c = ds.corrupted_entries[i, :]
    print(c)
    j = torch.nonzero(c).flatten()[0].item()
    x = ds[i][0][j]
    x_original = ds.rgb_cells[i][0][j]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x)
    plt.subplot(1, 2, 2)
    plt.imshow(x_original)
    plt.show()

##
# quick exploration of the channels to visually determine where expression is located within the cell

n_channels = ds[42][0].shape[0]
n_cells = 10

axes = plt.subplots(n_channels, n_cells, figsize=(2 * n_cells, 2 * n_channels))[
    1
].T.flatten()
for j in tqdm(range(n_cells)):
    ome, mask, is_perturbed = ds[j]
    numpy_mask = mask.squeeze(0).numpy()
    contours = measure.find_contours(numpy_mask, 0.4)
    for c in range(n_channels):
        k = j * n_channels + c
        ax = axes[k]
        p = "" if not is_perturbed[c] else " (perturbed)"
        ax.set(title=f"ds[{j}], ch {c}{p}")
        ax.imshow(ome[c, :, :].numpy())

        # show mask border
        # ax.imshow(numpy_mask, alpha=0.4, cmap=matplotlib.cm.gray)
        for contour in contours:
            orange = list(map(lambda x: x / 255, (255, 165, 0)))
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=orange)

plt.tight_layout()
plt.show()
##
# selected channels
d = {
    0: "subcellular",
    37: "subcellular",
    38: "subcellular",
    3: "boundary",
    4: "boundary",
    5: "boundary",
    10: "both",
    35: "both",
}
# plotting more cells
n_channels = len(d)
for channel, description in tqdm(d.items(), desc="channels"):
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.flatten()
    plt.suptitle(f"channel {channel} ({description})")
    for i, ax in enumerate(axes):
        ome, mask, is_perturbed = ds[i]
        numpy_mask = mask.squeeze(0).numpy()
        contours = measure.find_contours(numpy_mask, 0.4)
        p = "" if not is_perturbed[channel] else " (perturbed)"
        ax.set(title=f"ds[{i}], ch {channel}{p}")
        ax.imshow(ome[channel, :, :].numpy())

        # show mask border
        # ax.imshow(numpy_mask, alpha=0.4, cmap=matplotlib.cm.gray)
        for contour in contours:
            orange = list(map(lambda x: x / 255, (255, 165, 0)))
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=orange)

    plt.tight_layout()
    plt.show()

##
