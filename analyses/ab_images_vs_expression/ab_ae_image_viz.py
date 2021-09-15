import PIL
import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid

from data2 import quantiles_for_normalization


def get_image(loader, model, return_cells=False):
    all_originals = []
    all_reconstructed = []
    all_masks = []
    mask_color = torch.tensor([x / 255 for x in [254, 112, 31]]).float()
    red_color = torch.tensor([x / 255 for x in [255, 0, 0]]).float()
    new_size = (128, 128)
    upscale = torchvision.transforms.Resize(new_size, interpolation=PIL.Image.NEAREST)
    n = 15
    with torch.no_grad():
        batch = loader.__iter__().__next__()
        omes = batch[0]
        masks = batch[1]
        if len(batch) == 3:
            perturbed_entries = batch[2]
        else:
            perturbed_entries = None
        assert len(omes.shape) == 4
        assert len(omes) >= n
        data = omes[:n].to(model.device)
        masks_data = masks[:n].to(model.device)
        alpha_pred, beta_pred, mu, std, z = model.forward(data, masks_data)
    n_channels = data.shape[1]
    # I'm lazy
    full_mask = torch.tensor(
        [[mask_color.tolist() for _ in range(n_channels)] for _ in range(n_channels)]
    )
    full_mask = upscale(full_mask.permute(2, 0, 1))
    all_original_c = {c: [] for c in range(n_channels)}
    all_original_masked_c = {c: [] for c in range(n_channels)}
    all_reconstructed_c = {c: [] for c in range(n_channels)}
    all_reconstructed_masked_c = {c: [] for c in range(n_channels)}

    for i in range(n):
        original = data[i].cpu().permute(1, 2, 0) * quantiles_for_normalization
        alpha = alpha_pred[i].cpu().permute(1, 2, 0)
        beta = beta_pred[i].cpu().permute(1, 2, 0)
        dist = model.get_dist(alpha, beta)
        mean = dist.mean
        # r = alpha
        # p = 1 / (1 + beta)
        # # r_hat = pred[i].cpu().permute(1, 2, 0)
        # # p = torch.sigmoid(model.negative_binomial_p_logit).cpu().detach()
        # # mean = model.negative_binomial_mean(r=r_hat, p=p)
        # mean = p * r / (1 - p)
        reconstructed = mean * quantiles_for_normalization

        a_original = original.amin(dim=(0, 1))
        b_original = original.amax(dim=(0, 1))
        m = masks_data[i].cpu().bool()
        mm = torch.squeeze(m, 0)
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

            mm_not = torch.logical_not(mm)
            assert torch.all(reconstructed[mm, :] >= 0.0)
            assert torch.all(reconstructed[mm, :] <= 1.0)
            reconstructed = torch.clamp(reconstructed, 0.0, 1.0)

            all_masks.append(mm)
            #### original_masked = original.clone()
            #### original_masked[mm_not, :] = mask_color
            #### reconstructed_masked = reconstructed.clone()
            #### reconstructed_masked[mm_not, :] = mask_color

            for c in range(n_channels):
                is_perturbed_entry = (
                        perturbed_entries is not None and perturbed_entries[i, c]
                )
                original_c = original[:, :, c]
                original_c = torch.stack([original_c] * 3, dim=2)

                reconstructed_c = reconstructed[:, :, c]
                reconstructed_c = torch.stack([reconstructed_c] * 3, dim=2)

                def f(t):
                    t = t.permute(2, 0, 1)
                    t = upscale(t)
                    return t

                def overlay_mask(t, is_perturbed_entry=False):
                    t = t.clone()
                    if not is_perturbed_entry:
                        color = mask_color
                    else:
                        color = red_color
                    t[mm_not, :] = color
                    return t

                a_original_c = original_c.amin(dim=(0, 1))
                b_original_c = original_c.amax(dim=(0, 1))
                reconstructed_flattened_c = torch.reshape(
                    reconstructed_c, (-1, reconstructed_c.shape[-1])
                )
                mask_flattened = mm.flatten()
                a_reconstructed_c = reconstructed_flattened_c[mask_flattened, :].amin(
                    dim=0
                )
                b_reconstructed_c = reconstructed_flattened_c[mask_flattened, :].amax(
                    dim=0
                )
                a_c = torch.min(a_original_c, a_reconstructed_c)
                b_c = torch.max(b_original_c, b_reconstructed_c)

                t = (original_c - a_c) / (b_c - a_c)
                all_original_c[c].append(f(t))
                all_original_masked_c[c].append(f(overlay_mask(t, is_perturbed_entry)))
                t = (reconstructed_c - a_c) / (b_c - a_c)
                all_reconstructed_c[c].append(f(t))
                all_reconstructed_masked_c[c].append(
                    f(overlay_mask(t, is_perturbed_entry))
                )

            original = upscale(original.permute(2, 0, 1))
            reconstructed = upscale(reconstructed.permute(2, 0, 1))
            #### original_masked = upscale(original_masked.permute(2, 0, 1))
            #### reconstructed_masked = upscale(reconstructed_masked.permute(2, 0, 1))

            all_originals.append(original)
            all_reconstructed.append(reconstructed)
            #### all_originals_masked.append(original_masked)
            #### all_reconstructed_masked.append(reconstructed_masked)
        else:
            all_originals.append(upscale(original.permute(2, 0, 1)))
            all_reconstructed.append(upscale(reconstructed.permute(2, 0, 1)))
            all_masks.append(torch.tensor(np.zeros(new_size, dtype=bool)))
            for c in range(n_channels):
                all_original_c[c].append(full_mask)
                all_reconstructed_c[c].append(full_mask)
                all_original_masked_c[c].append(full_mask)
                all_reconstructed_masked_c[c].append(full_mask)

    l = []  ####
    pixels = []
    for original, reconstructed, mask in zip(
            all_originals, all_reconstructed, all_masks
    ):
        pixels.extend(original.permute(1, 2, 0).reshape((-1, n_channels)))
        upscaled_mask = upscale(torch.unsqueeze(mask, 0))
        upscaled_mask = torch.squeeze(upscaled_mask, 0).bool()
        masked_reconstructed = reconstructed[:, upscaled_mask]
        pixels.extend(masked_reconstructed.permute(1, 0))
    from sklearn.decomposition import PCA

    all_pixels = torch.stack(pixels).numpy()
    reducer = PCA(3)
    reducer.fit(all_pixels)
    transformed = reducer.transform(all_pixels)
    a = np.min(transformed, axis=0)
    b = np.max(transformed, axis=0)
    all_originals_pca = []
    all_reconstructed_pca = []
    all_originals_pca_masked = []
    all_reconstructed_pca_masked = []

    def scale(x, a, b):
        x = np.transpose(x, (1, 2, 0))
        x = (x - a) / (b - a)
        x = np.transpose(x, (2, 0, 1))
        return x

    for original, reconstructed, mask in zip(
            all_originals, all_reconstructed, all_masks
    ):
        to_transform = original.permute(1, 2, 0).reshape((-1, n_channels)).numpy()
        pca = reducer.transform(to_transform)
        pca.shape = (original.shape[1], original.shape[2], 3)
        original_pca = np.transpose(pca, (2, 0, 1))
        original_pca = scale(original_pca, a, b)
        all_originals_pca.append(torch.tensor(original_pca))

        to_transform = reconstructed.permute(1, 2, 0).reshape((-1, n_channels)).numpy()
        pca = reducer.transform(to_transform)
        pca.shape = (reconstructed.shape[1], reconstructed.shape[2], 3)
        reconstructed_pca = np.transpose(pca, (2, 0, 1))
        reconstructed_pca = scale(reconstructed_pca, a, b)
        all_reconstructed_pca.append(torch.tensor(reconstructed_pca))

        mask = torch.logical_not(mask)
        upscaled_mask = upscale(torch.unsqueeze(mask, 0))
        upscaled_mask = torch.squeeze(upscaled_mask, 0).bool()

        original_pca_masked = original_pca.copy()
        original_pca_masked = original_pca_masked.transpose((1, 2, 0))
        original_pca_masked[upscaled_mask, :] = mask_color
        original_pca_masked = original_pca_masked.transpose((2, 0, 1))
        all_originals_pca_masked.append(torch.tensor(original_pca_masked))

        reconstructed_pca_masked = reconstructed_pca.copy()
        reconstructed_pca_masked = reconstructed_pca_masked.transpose((1, 2, 0))
        reconstructed_pca_masked[upscaled_mask, :] = mask_color
        reconstructed_pca_masked = reconstructed_pca_masked.transpose((2, 0, 1))
        all_reconstructed_pca_masked.append(torch.tensor(reconstructed_pca_masked))

    l.extend(
        all_originals_pca
        + all_reconstructed_pca
        + all_originals_pca_masked
        + all_reconstructed_pca_masked
    )

    for c in range(n_channels):
        l += (
                all_original_c[c]
                + all_reconstructed_c[c]
                + all_original_masked_c[c]
                + all_reconstructed_masked_c[c]
        )

    #### from sklearn.decomposition import PCA
    #### reducer = PCA(3)

    img = make_grid(l, nrow=n)
    if not return_cells:
        return img
    else:
        return img, all_original_c, all_reconstructed_c, all_masks
