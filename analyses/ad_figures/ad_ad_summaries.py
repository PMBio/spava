##

import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from analyses.aa_reconstruction_benchmark.aa_ad_reconstruction import Prediction, Space
from analyses.essentials import louvain_plot, scanpy_compute
from data2 import file_path

##
d0 = pickle.load(open(file_path("scvi_scores.pickle"), "rb"))
d1 = pickle.load(open(file_path("ah_scores.pickle"), "rb"))
d2 = pickle.load(open(file_path("baseline_scores.pickle"), "rb"))
d3 = pickle.load(open(file_path("conv_encoder_scores.pickle"), "rb"))
d4 = pickle.load(open(file_path("ai_gnn_vae_scores.pickle"), "rb"))
d5 = pickle.load(open(file_path("latent_lego_scores.pickle"), "rb"))
d = {**d0, **d1, **d2, **d3, **d4, **d5}
scores = {}
for k, v in d.items():
    print(k)
    p = Prediction(**v)
    if p.space != Space.scaled_mean:
        p = p.transform_to(Space.scaled_mean)
    d[k] = p
    n_channels = p.original.shape[-1]
    (
        original_non_perturbed_by_channel,
        reconstructed_zero_by_channel,
    ) = p._get_corrupted_entries_values_by_channel()

    ss = []
    for i in range(n_channels):
        original = original_non_perturbed_by_channel[i]
        imputed = reconstructed_zero_by_channel[i]
        s = np.mean(p.score_function(original, imputed))
        ss.append(s)
    ss = np.array(ss)
    scores[k] = ss


##
def get_df(columns: List[str]):
    a = [(k, v.mean()) for k, v in scores.items()]
    a = sorted(a, key=lambda x: x[1])
    sorter = dict(a)

    l = []
    for k, v in scores.items():
        if k not in columns:
            continue
        df = pd.DataFrame(v.reshape(1, -1))
        df_melt = df.melt(value_vars=[i for i in range(n_channels)])
        df_melt["name"] = k
        l.append(df_melt)
    df = pd.concat(l)
    df["sorter"] = df["name"].apply(lambda x: sorter[x])
    df.sort_values(by="sorter", inplace=True)
    df.rename(columns={"variable": "channel", "value": "score"}, inplace=True)
    return df


def scores_barplot(columns: List[str]):
    df = get_df(columns)

    plt.figure(figsize=(20, 4))
    sns.barplot(x="channel", y="score", hue="name", data=df)
    plt.show()


c0 = [
    "vanilla VAE",
    "scVI",
    "LatentLego",
    "ConvVAE expression",
    "ai_gnn_vae expression",
]
c1 = [
    "mean predictor (constant)",
    "linear model",
    "random forest",
    "simple neural network",
    "ConvVAE expression",
]
scores_barplot(c0)
scores_barplot(c1)

##
def scores_boxplots(columns: List[str], top=False):
    df = get_df(columns)
    means = df.groupby("name")["score"].mean()
    means = means.sort_values()
    means

    plt.figure(figsize=(3, 5))
    ax = sns.stripplot(
        x="name",
        y="score",
        data=df,
        # showmeans=True,
        # meanprops={
        #     "marker": "o",
        #     "markerfacecolor": "white",
        #     "markeredgecolor": "black",
        #     "markersize": "8",
        # },
    )
    ax.tick_params(axis="x", rotation=90)
    ax.scatter(np.arange(len(means)), means, s=400, c="k", zorder=10, marker="_")
    if top:
        ax.tick_params(labelbottom=False, labeltop=True)
    plt.tight_layout()
    plt.show()


scores_boxplots(c0)
scores_boxplots(c1, top=True)

##
import analyses.aa_reconstruction_benchmark.aa_ad_reconstruction
import importlib

importlib.reload(analyses.aa_reconstruction_benchmark.aa_ad_reconstruction)
from analyses.aa_reconstruction_benchmark.aa_ad_reconstruction import Prediction

for k, v in scores.items():
    f = lambda i: f"{np.argsort(v)[i]}: {v[np.argsort(v)[i]]}"
    # print(f(0))
    i = round(len(v) / 2)
    # print(f(i))
    # print(f(-1))
    indices = [np.argsort(v)[0], np.argsort(v)[i], np.argsort(v)[-1]]
    labels = ["best", "median", "worst"]
    p = d[k]
    (
        original_non_perturbed_by_channel,
        reconstructed_zero_by_channel,
    ) = p._get_corrupted_entries_values_by_channel()

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
        score = p.score_function(original, imputed)
        ax.set(title=f"{label}, score: {p.format_score(np.mean(score))}")
        if i == 0:
            ax.set(xlabel="original", ylabel="imputed")
        # if i > 2:
        #     break
    fig.suptitle(
        f"{p.name}, {p.space.name}, global score: "
        f"{p.format_score(np.mean(p.scores_perturbed))}"
    )
    plt.tight_layout()
    plt.show()
    plt.style.use("default")

##
k = "mean predictor (constant)"
v = scores[k]
np.argsort(v)
np.sort(v)
indices
p = d[k]
p.predictions_from_perturbed

plt.figure()
ax = plt.gca()
plt.show()
original.max()
imputed.max()
imputed.min()
v

##
"""
LATENT SPACE ANALYSIS
"""
d_scvi = pickle.load(open(file_path("latent_anndata_from_scvi.pickle"), "rb"))
d_ah = pickle.load(open(file_path("latent_anndata_from_ah_model.pickle"), "rb"))
d_conv_vae_expression = pickle.load(
    open(file_path("latent_anndata_from_conv_encoder.pickle"), "rb")
)
d_gnn_vae_expression = pickle.load(
    open(file_path("latent_anndata_from_ai_gnn_vae.pickle"), "rb")
)

##

# ii = reproducible_random_choice(len(adata0), 10000)
# bdata0 = adata0[ii].copy()
# bdata0 = adata0[ii].copy()
# bdata0 = adata0.copy()
# bdata1 = adata1.copy()
a_input_scvi = d_scvi["input"]
a_latent_scvi = d_scvi["latent"]
a_input_ah = d_ah["input"]
a_latent_ah = d_ah["latent"]
a_latent_conv_vae_expression = d_conv_vae_expression["latent"]
a_latent_gnn_vae_expression = d_gnn_vae_expression["latent"]
assert len(a_input_scvi) == len(a_latent_scvi)
assert len(a_input_ah) == len(a_latent_ah)
assert len(a_input_scvi) == len(a_latent_ah)

##
scanpy_compute(a_latent_scvi)
louvain_plot(a_latent_scvi, "scVI latent")
scanpy_compute(a_latent_ah)
louvain_plot(a_latent_ah, "vanilla VAE latent")

##
import analyses.ab_images_vs_expression.ab_aa_expression_latent_samples

importlib.reload(analyses.ab_images_vs_expression.ab_aa_expression_latent_samples)
from analyses.ab_images_vs_expression.ab_aa_expression_latent_samples import (
    compare_clusters,
)

compare_clusters(a_latent_scvi, a_latent_ah, 'latent clusters: "scVI" vs "vanilla VAE"')

##
scanpy_compute(a_input_scvi)
louvain_plot(a_input_scvi, "scVI input")
scanpy_compute(a_input_ah)
louvain_plot(a_input_ah, "vanilla VAE input")

##
compare_clusters(a_input_scvi, a_latent_scvi, '"scVI input"\nvs\n"scVI latent"')
compare_clusters(
    a_input_ah, a_latent_ah, '"vanilla VAE input"\nvs\n"vanilla VAE latent"'
)
compare_clusters(a_input_ah, a_input_scvi, '"vanilla VAE input"\nvs\n"scVI VAE input"')
compare_clusters(a_latent_ah, a_latent_scvi, '"vanilla VAE latent"\nvs\n"scVI latent"')

##
scanpy_compute(a_latent_conv_vae_expression)
louvain_plot(a_latent_conv_vae_expression, "ConvVAE expression")
compare_clusters(
    a_latent_ah,
    a_latent_conv_vae_expression,
    '"vanilla VAE latent"\nvs\n"ConvVAE to expression latent"',
)
compare_clusters(
    a_latent_scvi,
    a_latent_conv_vae_expression,
    '"scVI latent"\nvs\n"ConvVAE to expression latent"',
)

##
scanpy_compute(a_latent_gnn_vae_expression)
louvain_plot(a_latent_gnn_vae_expression, "GNN-VAE expression")
compare_clusters(
    a_latent_ah,
    a_latent_gnn_vae_expression,
    '"vanilla VAE latent"\nvs\n"GNN-VAE to expression latent"',
)
compare_clusters(
    a_latent_scvi,
    a_latent_gnn_vae_expression,
    '"scVI latent"\nvs\n"GNN-VAE to expression latent"',
)

##
"""
ELBO
"""
# ah model: 145
# conv to expression: 116
