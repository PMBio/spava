##
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import dill
from tqdm import tqdm
from typing import Union

from data2 import PerturbedRGBCells, PerturbedCellDataset, IndexInfo
from models.ah_expression_vaes_lightning import VAE as ExpressionVAE, get_loaders
from utils import reproducible_random_choice

COMPLETE_RUN = False
# COMPLETE_RUN = False
PERTURB_ENTIRE_CELLS = False

m = __name__ == "__main__"
##
if m:
    SPLIT = "validation"
    ds = PerturbedRGBCells(split=SPLIT)
    cells_ds = PerturbedCellDataset(split=SPLIT)

    ds.perturb()
    cells_ds.perturb()
    assert np.all(ds.corrupted_entries.numpy() == cells_ds.corrupted_entries.numpy())

    perturb_kwargs = {"perturb": True}
    if PERTURB_ENTIRE_CELLS:
        perturb_kwargs["perturb_entire_cells"] = True

##
if m:
    ii = IndexInfo(SPLIT)
    n = ii.filtered_ends[-1]
    random_indices = reproducible_random_choice(n, 10000)

##
# re-train the best model but by perturbing the dataset
# if m:
if m and False:
    # train the model on dilated masks using the hyperparameters from the best model for original expression
    from models.ah_expression_vaes_lightning import objective, ppp
    from data2 import file_path

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study_name = "no-name-fbdac942-b370-43af-a619-621755ee9d1f"
    ppp.PERTURB_PIXELS = False
    ppp.PERTURB_PIXELS_SEED = 42
    ppp.PERTURB_MASKS = False
    ppp.PERTURB = True
    study = optuna.load_study(
        study_name=study_name, storage="sqlite:///" + file_path("optuna_ah.sqlite")
    )
    objective(study.best_trial)
    sys.exit(0)

##
from analyses.ab_images_vs_expression.ab_aa_expression_latent_samples import (
    precompute,
    louvain_plot,
)

if m:
    # 133
    MODEL_CHECKPOINT = (
        "/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/expression_vae/version_145"
        "/checkpoints/last.ckpt"
    )
##
if m and COMPLETE_RUN:
    b0, b1 = precompute(
        data_loaders=get_loaders(**perturb_kwargs),
        expression_model_checkpoint=MODEL_CHECKPOINT,
        random_indices=random_indices,
        split=SPLIT,
    )
##
if m and COMPLETE_RUN:
    louvain_plot(b1, "expression (perturbed)")
    louvain_plot(b0, "latent (perturbed)")

##
if m:
    loader = get_loaders(**perturb_kwargs)[["train", "validation"].index(SPLIT)]
    loader_non_perturbed = get_loaders(perturb=False)[
        ["train", "validation"].index(SPLIT)
    ]
    data = loader.__iter__().__next__()
    data_non_perturbed = loader_non_perturbed.__iter__().__next__()
    i0, i1 = torch.where(data[-1] == 1)
    # perturbed data are zero, non perturbed data are ok
    print(data[0][i0, i1])
    print(data_non_perturbed[0][i0, i1])
    # just a hash
    h = torch.sum(torch.cat(torch.where(loader.dataset.corrupted_entries == 1)))
    print(
        "corrupted entries hash:",
        h,
    )

##
if m:
    expression_vae = ExpressionVAE.load_from_checkpoint(MODEL_CHECKPOINT)

    debug_i = 0
    print("merging expressions and computing embeddings... ", end="")
    all_mu = []
    all_expression = []
    all_a = []
    all_b = []
    all_is_perturbed = []
    all_expression_non_perturbed = []
    for data, data_non_perturbed in tqdm(
        zip(loader, loader_non_perturbed),
        desc="embedding expression",
        total=len(loader),
    ):
        expression, _, is_perturbed = data
        expression_non_perturbed, _, _ = data_non_perturbed
        a, b, mu, std, z = expression_vae(expression)
        all_mu.append(mu)
        all_expression.append(expression)
        all_a.append(a)
        all_b.append(b)
        all_is_perturbed.append(is_perturbed)
        all_expression_non_perturbed.append(expression_non_perturbed)
        if debug_i < 5:
            # perturbed entries
            i0, i1 = torch.where(data[-1] == 1)
            # non perturbed entries
            j0, j1 = torch.where(data[-1] == 0)
            assert torch.isclose(torch.sum(expression[i0, i1]), torch.tensor([0.0]))
            assert torch.all(expression[j0, j1] == expression_non_perturbed[j0, j1])
        debug_i += 1

    mus = torch.cat(all_mu, dim=0)
    expressions = torch.cat(all_expression, dim=0)
    expressions_non_perturbed = torch.cat(all_expression_non_perturbed, dim=0)
    a_s = torch.cat(all_a, dim=0)
    b_s = torch.cat(all_b, dim=0)
    are_perturbed = torch.cat(all_is_perturbed, dim=0)
    reconstructed = expression_vae.expected_value(a_s, b_s)

##
from enum import IntEnum
from typing import Callable
from data2 import quantiles_for_normalization

from data2 import AreaFilteredDataset
from utils import memory
from tqdm import tqdm
import numpy as np


@memory.cache
def f_xqoifaowi():
    d = {}
    for split in ["train", "validation", "test"]:
        area_ds = AreaFilteredDataset(split)

        l = []
        for x in tqdm(area_ds, desc="merging"):
            l.append(x)
        areas = np.concatenate(l, axis=0)
        d[split] = areas
    return d


areas = f_xqoifaowi()
##
Space = IntEnum("Space", "raw_sum raw_mean asinh_sum asinh_mean scaled_mean", start=0)

from_to = [[None for _ in range(len(Space))] for _ in range(len(Space))]


def f_set(space0: Space, space1: Space, transformation: Callable):
    from_to[space0.value][space1.value] = transformation


def f_get(space0: Space, space1: Space):
    return from_to[space0.value][space1.value]


f_set(Space.raw_sum, Space.asinh_sum, lambda x, a: np.arcsinh(x))
f_set(Space.raw_mean, Space.asinh_mean, lambda x, a: np.arcsinh(x))
f_set(Space.asinh_mean, Space.scaled_mean, lambda x, a: x / quantiles_for_normalization)
f_set(Space.raw_sum, Space.raw_mean, lambda x, a: x / a)

f_set(Space.asinh_sum, Space.raw_sum, lambda x, a: np.sinh(x))
f_set(Space.asinh_mean, Space.raw_mean, lambda x, a: np.sinh(x))
f_set(Space.scaled_mean, Space.asinh_mean, lambda x, a: x * quantiles_for_normalization)
f_set(Space.raw_mean, Space.raw_sum, lambda x, a: x * a)

from typing import List


def find_path(from_node: int, to_node: int):
    def dfs(from_node: int, to_node: int, visited: List[int], path: List[int]):
        if from_node == to_node:
            return True
        for j, to in enumerate(from_to[from_node]):
            if to is not None:
                if not visited[j]:
                    visited[j] = True
                    b = dfs(from_node=j, to_node=to_node, visited=visited, path=path)
                    if b:
                        path.append(j)
                        return True
        return False

    visited = [False for _ in range(len(Space))]
    path = []
    dfs(from_node=from_node, to_node=to_node, visited=visited, path=path)
    path.append(from_node)
    return list(reversed(path))


test_path = find_path(from_node=Space.scaled_mean.value, to_node=Space.asinh_sum.value)
assert test_path == [4, 3, 1, 0, 2], test_path


def transform(
    x: np.ndarray, from_space: Space, to_space: Space, split: str
) -> np.ndarray:
    a = areas[split]
    assert len(a) == len(x)
    path = find_path(from_node=from_space.value, to_node=to_space.value)
    for i in range(len(path) - 1):
        start, end = path[i], path[i + 1]
        s, e = Space(start), Space(end)
        f = f_get(s, e)
        print(f"applying transformation from {s.name} to {e.name}")
        x = f(x, a)
    return x


# transform(x=np.random.rand(218618, 39), from_space=Space.scaled_sum, to_space=Space.scaled_mean, split='test')


# f_set(Space.raw, Space.asinh, lambda x: np.arcsinh(x))
# f_set(Space.asinh, Space.raw, lambda x: np.sinh(x))
# f_set(Space.asinh, Space.scaled, lambda x: x / quantiles_for_normalization)
# f_set(Space.scaled, Space.asinh, lambda x: x * quantiles_for_normalization)
# f_set(
#     Space.raw,
#     Space.scaled,
#     lambda x: f_get(Space.asinh, Space.scaled)(f_get(Space.raw, Space.asinh)(x)),
# )
# f_set(
#     Space.scaled,
#     Space.raw,
#     lambda x: f_get(Space.asinh, Space.raw)(f_get(Space.scaled, Space.asinh)(x)),
# )

import math
from scipy.stats import t as t_dist


class Prediction:
    def __init__(
        self,
        original: np.ndarray,
        corrupted_entries: np.ndarray,
        predictions_from_perturbed: np.ndarray,
        space: Union[Space, int],
        name: str,
        split: str,
    ):
        self.original = original
        self.corrupted_entries = corrupted_entries
        self.predictions_from_perturbed = predictions_from_perturbed
        if type(space) == int:
            space = Space(space)
        self.space = space
        self.name = name
        self.split = split
        self.scores_non_perturbed = None
        self.scores_perturbed = None
        assert self.original.shape == self.corrupted_entries.shape
        assert self.original.shape == self.predictions_from_perturbed.shape

    def transform_to(self, space: Space) -> Prediction:
        original_transformed = transform(
            x=self.original,
            from_space=self.space,
            to_space=space,
            split=self.split,
        )
        predictions_from_perturbed_transformed = transform(
            x=self.predictions_from_perturbed,
            from_space=self.space,
            to_space=space,
            split=self.split,
        )
        p = Prediction(
            original=original_transformed,
            corrupted_entries=self.corrupted_entries,
            predictions_from_perturbed=predictions_from_perturbed_transformed,
            space=space,
            name=self.name,
            split=self.split,
        )
        return p

    def score_function(self, original, reconstructed):
        return np.square(original - reconstructed)

    def format_score(self, score):
        return f"{score:0.3f}"

    def compute_scores(self):
        ce = self.corrupted_entries
        ne = np.logical_not(self.corrupted_entries)

        uu0 = self.predictions_from_perturbed[ce]
        uu1 = self.original[ce]

        vv0 = self.predictions_from_perturbed[ne]
        vv1 = self.original[ne]

        self.scores_perturbed = self.score_function(uu0, uu1)
        self.scores_non_perturbed = self.score_function(vv0, vv1)

    def check_scores_defined(self, test=False):
        b = self.scores_perturbed is not None and self.scores_non_perturbed is not None
        if test:
            return b
        else:
            assert b

    def plot_scores(self):
        self.check_scores_defined()

        s = self.scores_perturbed
        t = self.scores_non_perturbed

        # corrupted entries, normal entries
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(s)
        m_s = np.mean(s)
        plt.title(f"scores for imputed entries\nmean: {self.format_score(m_s)}")
        plt.yscale("log")

        plt.subplot(1, 2, 2)
        plt.hist(t)
        m_t = np.mean(t)
        plt.title(f"control: normal entries\nmean: {self.format_score(m_t)}")
        plt.yscale("log")

        fig.suptitle(f"{self.name}: abs(original vs predicted)")
        plt.tight_layout()
        plt.show()

        return m_s, s

    @staticmethod
    def welch_t_test(abs_values0: np.ndarray, abs_values1: np.ndarray):
        n = min(len(abs_values0), len(abs_values1))
        # s and t here are in general different from the s and t in the function compute_and_plot_scores
        s = np.random.choice(abs_values0, n, replace=False)
        t = np.random.choice(abs_values1, n, replace=False)

        # Welch's t test
        sem_s = s.std() / math.sqrt(len(s))
        sem_t = t.std() / math.sqrt(len(t))
        sed = math.sqrt(sem_s ** 2 + sem_t ** 2)
        t_statistic = (s.mean() - t.mean()) / sed

        # degrees of freedom from the Welch-Satterthwaite equation
        delta = s - t
        sem_delta = delta.std() / math.sqrt(len(delta))
        df = sem_delta ** 4 / (
            (len(s) - 1) ** -1 * sem_s ** 4 + (len(t) - 1) ** -1 * sem_t ** 4
        )
        # print(f"df = {df}")

        p_value = 1 - t_dist.cdf(t_statistic, df)
        print(f"welch's t test: p_value = {p_value}")

    @staticmethod
    def plot_imputation(imputed, original, ax):  # , zeros, i, j, ix, xtext):
        from scipy.stats import kde
        import time

        # all_index = i[ix], j[ix]
        # x, y = imputed[all_index], original[all_index]
        #
        # x = x[zeros[all_index] == 0]
        # y = y[zeros[all_index] == 0]
        #
        q = 0.9
        # cutoff = max(np.quantile(original, q), np.quantile(imputed, q))
        cutoff = np.quantile(original, q) * 1.2
        # debug stuff
        # print(f"cutoff = {cutoff}")
        # print(f"len(original) = {len(original)}")
        # print(f"original[:3] = {original[:3]}")
        # print(f"np.quantile(original, q) = {np.quantile(original, q)}")
        # sys.exit(1)
        mask = imputed < cutoff
        imputed = imputed[mask]
        original = original[mask]

        mask = original < cutoff
        imputed = imputed[mask]
        original = original[mask]

        l = np.minimum(imputed.shape[0], original.shape[0])

        if l == 0:
            return

        assert len(imputed) == len(original)
        imputed = imputed[:l]
        original = original[:l]

        # data = np.vstack([x, y])
        data = np.vstack([imputed, original])

        ax.set_xlim([0, cutoff])
        ax.set_ylim([0, cutoff])

        nbins = 50

        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        try:
            k = kde.gaussian_kde(data)
        except ValueError as e:
            print(data)
            print(data.shape)
            print(data.min())
            print(data.max())
            print(cutoff)
            print(original.max())
            raise e
        except np.linalg.LinAlgError as e:
            if e.args == ("singular matrix",):
                print("warning: singular matrix when calling kde.gaussian_kde()")
                return
            else:
                raise e
        xi, yi = np.mgrid[0 : cutoff : nbins * 1j, 0 : cutoff : nbins * 1j]

        start = time.time()
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # print(f"evaluating the kernel on the mesh: {time.time() - start}")

        ax.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap="Reds", shading="gouraud")

        a, _, _, _ = np.linalg.lstsq(original[:, np.newaxis], imputed, rcond=None)
        l = np.linspace(0, cutoff)
        ax.plot(l, a * l, color="black")

        # A = np.vstack([original, np.ones(len(original))]).T
        # aa, _, _, _ = np.linalg.lstsq(A, imputed, rcond=None)
        # ax.plot(l, aa[0] * l + aa[1], color="red")

        ax.plot(l, l, color="black", linestyle=":")

    def _get_corrupted_entries_values_by_channel(self):
        if not self.check_scores_defined(test=True):
            self.compute_scores()

        n_channels = self.original.shape[-1]
        original_non_perturbed_by_channel = []
        reconstructed_zero_by_channel = []
        for i in range(n_channels):
            x = self.original[:, i][self.corrupted_entries[:, i]]
            original_non_perturbed_by_channel.append(x)
            x = self.predictions_from_perturbed[:, i][self.corrupted_entries[:, i]]
            reconstructed_zero_by_channel.append(x)
        return original_non_perturbed_by_channel, reconstructed_zero_by_channel

    # def plot_reconstruction(original_non_perturbed, reconstructed_zero, n_channels):
    def plot_reconstruction(self):
        n_channels = self.original.shape[-1]
        (
            original_non_perturbed_by_channel,
            reconstructed_zero_by_channel,
        ) = self._get_corrupted_entries_values_by_channel()

        from matplotlib.lines import Line2D

        d = 3
        fig, axes = plt.subplots(5, 8, figsize=(8 * d, 5 * d))
        axes = axes.flatten()

        custom_lines = [
            Line2D([0], [0], color="black", linestyle=":", lw=1),
            Line2D([0], [0], color="black", lw=1),
            # Line2D([0], [0], color="red", lw=1),
        ]

        axes[0].legend(
            custom_lines,
            [
                "identity",
                "linear model",
                # "affine model"
            ],
            loc="center",
        )
        axes[0].set_axis_off()

        for i in tqdm(range(n_channels), desc="channels"):
            ax = axes[i + 1]
            original = original_non_perturbed_by_channel[i]
            imputed = reconstructed_zero_by_channel[i]
            Prediction.plot_imputation(
                original=original,
                imputed=imputed,
                ax=ax,
            )
            score = self.score_function(original, imputed)
            ax.set(title=f"ch {i}, score: {self.format_score(np.mean(score))}")
            if i == 0:
                ax.set(xlabel="original", ylabel="imputed")
            # if i > 2:
            #     break
        fig.suptitle(
            f"{self.name}, {self.space.name}, global score: "
            f"{self.format_score(np.mean(self.scores_perturbed))}"
        )
        plt.tight_layout()
        plt.show()

    def plot_scores(self):
        n_channels = self.original.shape[-1]
        (
            original_non_perturbed_by_channel,
            reconstructed_zero_by_channel,
        ) = self._get_corrupted_entries_values_by_channel()

        scores = []
        for i in range(n_channels):
            score = np.mean(
                self.score_function(
                    original_non_perturbed_by_channel[i],
                    reconstructed_zero_by_channel[i],
                ).item()
            )
            scores.append(score)
        plt.figure()
        plt.bar(np.arange(n_channels), np.array(scores))
        plt.title(
            f"{self.name}, reconstruction scores, global score: {self.format_score(np.mean(self.scores_perturbed))}"
        )
        plt.xlabel("channel")
        plt.ylabel("score")
        plt.show()


def compare_predictions(p0: Prediction, p1: Prediction, target_space: Space):
    q0 = p0.transform_to(target_space)
    q1 = p1.transform_to(target_space)
    m0, s0 = q0.compute_and_plot_scores()
    m1, s1 = q1.compute_and_plot_scores()
    Prediction.welch_t_test(s0, s1)


##
# if m:
#     n_channels = expressions.shape[1]
#
#     original_non_perturbed = []
#     reconstructed_zero = []
#     for i in range(n_channels):
#         x = expressions_non_perturbed[:, i][are_perturbed[:, i]]
#         original_non_perturbed.append(x)
#         x = reconstructed[:, i][are_perturbed[:, i]]
#         reconstructed_zero.append(x)

##
if m:
    kwargs = dict(
        original=expressions_non_perturbed.cpu().numpy(),
        corrupted_entries=are_perturbed.cpu().numpy(),
        predictions_from_perturbed=reconstructed.detach().cpu().numpy(),
        space=Space.scaled_mean.value,
        name="ah_expression",
        split="validation",
    )
    ah_predictions = Prediction(
        **kwargs
    )

    ah_predictions.plot_reconstruction()
    # ah_predictions.plot_scores()
#
if m and False:
    p = ah_predictions.transform_to(Space.raw_sum)
    p.name = "ah_expression raw"
    p.plot_reconstruction()
    # p.plot_scores()

##
from data2 import file_path
import pickle

if m:
    d = {"vanilla VAE": kwargs}
    pickle.dump(d, open(file_path("ah_scores.pickle"), "wb"))