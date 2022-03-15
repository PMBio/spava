from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist
from tqdm.auto import tqdm
import warnings


class Prediction:
    def __init__(
        self,
        original: np.ndarray,
        corrupted_entries: np.ndarray,
        predictions_from_perturbed: np.ndarray,
        name: str,
    ):
        self.original = original
        self.corrupted_entries = corrupted_entries
        self.predictions_from_perturbed = predictions_from_perturbed
        self.name = name
        self.scores_non_perturbed = None
        self.scores_perturbed = None
        assert self.original.shape == self.corrupted_entries.shape
        assert self.original.shape == self.predictions_from_perturbed.shape

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

    def plot_scores_global(self):
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
    def crop(imputed, original):

        # all_index = i[ix], j[ix]
        # x, y = imputed[all_index], original[all_index]
        #
        # x = x[zeros[all_index] == 0]
        # y = y[zeros[all_index] == 0]
        #
        q = 0.9
        if len(imputed) == 0:
            return imputed, original, None
        cutoff = np.quantile(original, q) * 1.5
        # cutoff = max(original.max(), imputed.max())

        # cutoff = max(np.quantile(original, q), np.quantile(imputed, q))
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
            raise ValueError("No points found")

        assert len(imputed) == len(original)
        imputed = imputed[:l]
        original = original[:l]
        return imputed, original, cutoff

    @staticmethod
    def plot_lines(imputed, original, cutoff, color, ax):
        a, _, _, _ = np.linalg.lstsq(original[:, np.newaxis], imputed, rcond=None)
        l = np.linspace(0, cutoff)
        ax.plot(l, a * l, color=color)

        # A = np.vstack([original, np.ones(len(original))]).T
        # aa, _, _, _ = np.linalg.lstsq(A, imputed, rcond=None)
        # ax.plot(l, aa[0] * l + aa[1], color="red")

        ax.plot(l, l, color=color, linestyle=":", alpha=0.5)

    @staticmethod
    def plot_imputation(imputed, original, ax):  # , zeros, i, j, ix, xtext):
        if len(imputed) == 0:
            return
        try:
            imputed, original, cutoff = Prediction.crop(imputed, original)
        except ValueError as e:
            # print(e.args)
            if str(e) == "No points found":
                print(str(e))
                return
        from scipy.stats import gaussian_kde
        import time

        # data = np.vstack([x, y])
        data = np.vstack([imputed, original])

        ax.set_xlim([0, cutoff])
        ax.set_ylim([0, cutoff])

        nbins = 50

        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        try:
            k = gaussian_kde(data)
        except ValueError as e:
            if str(e) == "array must not contain infs or NaNs":
                return
            print(data)
            print(data.shape)
            print(data.min())
            print(data.max())
            print(cutoff)
            print(original.max())
            raise e
        except np.linalg.LinAlgError as e:
            if e.args == ("singular matrix",):
                warnings.warn("singular matrix when calling kde.gaussian_kde()")
                return
            elif str(e) == '2-th leading minor of the array is not positive definite':
                warnings.warn(str(e))
                return
            else:
                print(str(e))
                raise e
        xi, yi = np.mgrid[0 : cutoff : nbins * 1j, 0 : cutoff : nbins * 1j]

        start = time.time()
        try:
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        except np.linalg.LinAlgError as e:
            if str(e) == "Matrix is not positive definite":
                warnings.warn(str(e))
                return
            else:
                print(str(e))
                raise e
        # print(f"evaluating the kernel on the mesh: {time.time() - start}")

        ax.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap="Reds", shading="gouraud")
        Prediction.plot_lines(imputed, original, cutoff, "black", ax)

    @staticmethod
    def plot_imputation2(imputed, original, ax):
        if len(imputed) == 0:
            return
        try:
            imputed, original, cutoff = Prediction.crop(imputed, original)
        except ValueError as e:
            # print(e.args)
            if str(e) == "No points found":
                print(str(e))
                return
        ax.set_xlim([0, cutoff])
        ax.set_ylim([0, cutoff])

        import datashader as ds
        import pandas as pd

        # plt.figure()
        df = pd.DataFrame(dict(x=original, y=imputed))
        w = 30
        h = 30
        k = 2
        dpi = 100

        x_range = (df.x.min(), df.x.max() + 0.1)
        y_range = (df.y.min(), df.y.max() + 0.1)
        # plt.figure(figsize=(w / dpi * k, h / dpi * k), dpi=dpi)
        cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_width=w, plot_height=h)
        agg = cvs.points(df, "x", "y")
        import matplotlib.cm

        img = ds.transfer_functions.shade(
            agg, cmap=matplotlib.cm.viridis
        )  # , cmap=["white", "black"])
        import PIL

        ax.imshow(
            img.to_pil()
            .transpose(PIL.Image.ROTATE_180)
            .transpose(PIL.Image.FLIP_LEFT_RIGHT)
            .transpose(PIL.Image.FLIP_TOP_BOTTOM),
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        )
        embl_green = np.array([6, 159, 77]) / 255
        color = embl_green
        color = "w"
        Prediction.plot_lines(imputed, original, cutoff, color, ax)

        # def get_n(a, b):
        #     a, b = min(a, b), max(a, b)
        #     n = np.ceil(-np.log10(b - a)).astype(int)
        #     print(a, b, n)
        #     return max(1, n)
        #
        # def get_f(n):
        #     # if n == 0:
        #     #     return 'd'
        #     # else:
        #     return f"0.{n}f"
        #
        # n = get_n(x_range[0], x_range[1])
        # f = get_f(n)
        # print(f)
        # x_ticklabels_values = np.linspace(x_range[0], x_range[1], 5)
        # x_ticklabels = list(map(lambda x: f'{x:{f}}', x_ticklabels_values))
        # x_ticks = [w * (x - x_range[0]) / (x_range[1] - x_range[0]) for x in x_ticklabels_values]
        #
        # n = get_n(y_range[0], y_range[1])
        # f = get_f(n)
        # print(f)
        # y_ticklabels_values = np.linspace(y_range[0], y_range[1], 5)
        # y_ticklabels = list(map(lambda y: f'{y:{f}}', y_ticklabels_values))
        # y_ticks = [h * (1 - (y - y_range[0]) / (y_range[1] - y_range[0])) for y in y_ticklabels_values]
        #
        # ax.set_xticks(x_ticks)
        # ax.set_xticklabels(x_ticklabels)
        # # ax.set_xlabel('protein expression')
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_ticklabels)
        # # ax.set_ylabel('ECDF value')
        # # plt.style.use('default')

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
        cols = 8
        rows = n_channels // cols + int(n_channels % cols != 0)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * d, rows * d))
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
            f"{self.name}, global score: "
            f"{self.format_score(np.mean(self.scores_perturbed))}"
        )
        plt.tight_layout()
        plt.show()

    def plot_scores(self, hist=False):
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
                )
            )
            scores.append(score)
        plt.figure()
        if hist:
            plt.hist(np.array(scores), bins=50)
            plt.xlabel("score")
            plt.ylabel("count")
            plt.yscale("log")
        else:
            plt.bar(np.arange(n_channels), np.array(scores))
            plt.xlabel("channel")
            plt.ylabel("score")
        plt.title(
            f"{self.name}, reconstruction scores, global score: {self.format_score(np.mean(self.scores_perturbed))}"
        )

        plt.show()

    def plot_summary(self):
        n_channels = self.original.shape[-1]
        (
            original_non_perturbed_by_channel,
            reconstructed_zero_by_channel,
        ) = self._get_corrupted_entries_values_by_channel()

        ss = []
        for i in range(n_channels):
            original = original_non_perturbed_by_channel[i]
            imputed = reconstructed_zero_by_channel[i]
            s = np.mean(self.score_function(original, imputed))
            ss.append(s)
        v = np.array(ss)
        f = lambda i: f"{np.argsort(v)[i]}: {v[np.argsort(v)[i]]}"
        # print(f(0))
        i = round(len(v) / 2)
        # print(f(i))
        # print(f(-1))
        indices = [np.argsort(v)[0], np.argsort(v)[i], np.argsort(v)[-1]]
        labels = ["best", "median", "worst"]

        from matplotlib.lines import Line2D

        dd = 3
        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 4, figsize=(4 * dd, 1 * dd))
        axes = axes.flatten()

        custom_lines = [
            Line2D([0], [0], color="black", linestyle=":", lw=1, c="w"),
            Line2D([0], [0], color="black", lw=1, c="w"),
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

        for i in tqdm(range(3), desc="channels"):
            ax = axes[i + 1]
            idx = indices[i]
            label = labels[i]
            original = original_non_perturbed_by_channel[idx]
            imputed = reconstructed_zero_by_channel[idx]
            # if False:
            if True:
                Prediction.plot_imputation2(
                    original=original,
                    imputed=imputed,
                    ax=ax,
                )
            else:
                Prediction.plot_imputation(
                    original=original,
                    imputed=imputed,
                    ax=ax,
                )
            score = self.score_function(original, imputed)
            ax.set(title=f"{label}, score: {self.format_score(np.mean(score))}")
            if i == 0:
                ax.set(xlabel="original", ylabel="imputed")
            # if i > 2:
            #     break
        fig.suptitle(
            f"{self.name}, global score: "
            f"{self.format_score(np.mean(self.scores_perturbed))}"
        )
        plt.tight_layout()
        plt.show()
        plt.style.use("default")


if __name__ == "__main__":
    x = np.random.rand(10000, 30)
    import torch
    from torch.distributions import Bernoulli

    dist = Bernoulli(probs=0.1)
    shape = x.shape
    state = torch.get_rng_state()
    torch.manual_seed(0)
    corrupted_entries = dist.sample(shape).bool().numpy()
    torch.set_rng_state(state)

    x_perturbed = x.copy()
    eps = np.random.rand(np.sum(corrupted_entries)) / 10
    x_perturbed[corrupted_entries] = x[corrupted_entries] + eps
    p = Prediction(
        original=x,
        corrupted_entries=corrupted_entries,
        predictions_from_perturbed=x_perturbed,
        name="test",
    )
    p.plot_reconstruction()
    p.plot_scores()
    p.plot_summary()
    print("ooo")
