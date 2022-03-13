##
from __future__ import annotations

import shutil
import scvi
import scanpy as sc
import torch

import numpy as np
from tqdm import tqdm
import anndata as ad
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import get_execute_function, file_path, memory, reproducible_random_choice
from datasets.imc_data import all_processed_smu

e_ = get_execute_function()

# from analyses.essentials import louvain_plot

# COMPLETE_RUN = True
COMPLETE_RUN = False
N_EPOCHS_KL_WARMUP = 3
N_EPOCHS = 10


##
if e_() and False:
    # proxy for the DKFZ network
    # https://stackoverflow.com/questions/34576665/setting-proxy-to-urllib-request-python3
    os.environ["HTTP_PROXY"] = "http://193.174.53.86:80"
    os.environ["HTTPS_PROXY"] = "https://193.174.53.86:80"

    # to have a look at an existing dataset
    import scvi.data

    data = scvi.data.pbmc_dataset()
    data

##
if e_() or True:
    from datasets.imc_data_transform_utils import joblib_get_merged_areas_per_split
    areas = joblib_get_merged_areas_per_split(ignore=e_())

    # TODO: train, validation and test
    @memory.cache
    def joblib_merge_sum_data():
        l0 = []
        l1 = []
        for i, s in enumerate(all_processed_smu()):
            x = s['imc']['sum'].X
            l0.append(x)
            l1.extend([i] * len(x))
        return l0, l1

    l0, l1 = joblib_merge_sum_data()
    raw = np.concatenate(l0, axis=0)
    raw = np.round(raw)
    raw = raw.astype(int)
    donor = np.array(l1)
    a = ad.AnnData(raw)

# ##
# if e_():
#     s = pd.Series(donor, index=a.obs.index)
#     a.obs["batch"] = s
#
# ##
# if e_():
#     scvi.data.setup_anndata(
#         a,
#         # this is probably meaningless (if not even penalizing) for unseen data as the batches are different
#         # categorical_covariate_keys=["batch"],
#     )
#     a
#
# ##
# if e_():
#     # TRAIN = True
#     TRAIN = False
#     if TRAIN:
#         # vae = VAE(gene_dataset.nb_genes)
#         # trainer = UnsupervisedTrainer(
#         #     vae,
#         #     gene_dataset,
#         #     train_size=0.90,
#         #     use_cuda=use_cuda,
#         #     frequency=5,
#         # )
#         # []:
#         # trainer.train(n_epochs=n_epochs, lr=lr)
#         model = scvi.model.SCVI(a)
#
# ##
#
# # the following code, as it is, doesn't work
# #     logger = TensorBoardLogger(save_dir=file_path("checkpoints"), name="scvi")
# #     BATCH_SIZE = 128
# #     indices = np.random.choice(len(a), BATCH_SIZE * 20, replace=False)
# #
# #     train_loader_batch = DataLoader(
# #         a.X[indices, :],
# #         batch_size=BATCH_SIZE,
# #         num_workers=4,
# #         pin_memory=True,
# #     )
# #     model.train(train_size=1., logger=logger, val_dataloaders=train_loader_batch)
# #     model.__dict__
# if e_():
#     if TRAIN:
#         model.train(
#             train_size=1.0, n_epochs=N_EPOCHS, n_epochs_kl_warmup=N_EPOCHS_KL_WARMUP
#         )
#         f = file_path("scvi_model.scvi")
#         if os.path.isdir(f):
#             shutil.rmtree(f)
#         model.save(f)
#     else:
#         model = scvi.model.SCVI.load(file_path("scvi_model.scvi"), adata=a)
#     print(model.get_elbo())
#
# ##
# if e_():
#     z = model.get_latent_representation()
#     a.shape
#     z.shape
#     b = ad.AnnData(z)
#     random_indices = reproducible_random_choice(len(a), 10000)
#     aa = a[random_indices]
#     bb = b[random_indices]
#
#
# def scanpy_compute(an: ad.AnnData):
#     sc.tl.pca(an)
#     print("computing neighbors... ", end="")
#     sc.pp.neighbors(an)
#     print("done")
#     print("computing umap... ", end="")
#     sc.tl.umap(an)
#     print("done")
#     print("computing louvain... ", end="")
#     sc.tl.louvain(an)
#     print("done")
#
#
# ##
# if m and COMPLETE_RUN:
#     scanpy_compute(aa)
#     sc.pl.pca(aa, title="pca, raw data (sum)")
#     ##
#     # plt.rcParams.update({'font.size': 22})
#     louvain_plot(aa, "UMAP with Louvain clusters, raw data (sum)")
#     # plt.figure(figsize=(10, 10))
#     # ax = plt.gca()
#     # plt.scatter(aa.obsm['X_umap'][:, 0], aa.obsm['X_umap'][:, 1])
#     # # sc.pl.umap(aa, color="louvain", title="umap with louvain, scvi latent (sum)", ax=ax)
#     # plt.tight_layout()
#     # plt.show()
#     # plt.rcParams.update({'font.size': 10})
#
# ##
# if m and COMPLETE_RUN:
#     scanpy_compute(bb)
#     sc.pl.pca(bb, title="pca, raw data (sum)")
#     # sc.pl.umap(bb, color="louvain", title="umap with louvain, scvi latent (sum)")
#     louvain_plot(bb, "UMAP with Louvain clusters, scvi latent (sum)")
#
# ##
#
#
# if m and COMPLETE_RUN:
#     compare_clusters(aa, bb, description='"raw data (sum)" vs "scvi latent"')
#     compute_knn(aa)
#     compute_knn(bb)
#     nearest_neighbors(
#         nn_from=aa, plot_onto=bb, title='nn from "raw data (sum)" to "scvi latent"'
#     )
#
# ##
# if e_():
#     ds = SumFilteredDataset("validation")
#
#     @memory.cache
#     def f_ncqoi3faoj(ds):
#         l0 = []
#         l1 = []
#         for i, x in enumerate(tqdm(ds, "merging")):
#             l0.append(x)
#             l1.extend([i] * len(x))
#         return l0, l1
#
#     l0, l1 = f_ncqoi3faoj(ds)
#     raw = np.concatenate(l0, axis=0)
#     donor = np.array(l1)
#     a_val = ad.AnnData(raw)
#
# ##
# if e_():
#     # note that here with are embedding without the batch information; if you want to look at batches it does not make
#     # sense to use another set except to the training one, since the train/val/test split is done by patient first
#     scvi.data.setup_anndata(
#         a_val,
#     )
#     z_val = model.get_latent_representation(a_val)
#     b_val = ad.AnnData(z_val)
#     random_indices_val = reproducible_random_choice(len(a_val), 10000)
#     aa_val = a_val[random_indices_val].copy()
#     bb_val = b_val[random_indices_val].copy()
#
# ##
# if m and COMPLETE_RUN:
#     scanpy_compute(aa_val)
#     scanpy_compute(bb_val)
#
# ##
# if m and COMPLETE_RUN:
#     sc.pl.pca(aa_val, title="pca, raw data (sum); validation set")
#     sc.pl.umap(
#         aa_val,
#         color="louvain",
#         title="umap with louvain, scvi latent (sum); valiation set",
#     )
#     sc.pl.pca(bb_val, title="pca, raw data (sum); valiation set")
#     sc.pl.umap(
#         bb_val,
#         color="louvain",
#         title="umap with louvain, scvi latent (sum); valiation set",
#     )
#
# ##
# if m and COMPLETE_RUN:
#     merged = ad.AnnData.concatenate(
#         bb, bb_val, batch_categories=["train", "validation"]
#     )
#     scanpy_compute(merged)
#     plt.figure()
#     ax = plt.gca()
#     sc.pl.umap(merged, color="batch", ax=ax, show=False)
#     plt.tight_layout()
#     plt.show()
#
# ##
# if e_():
#     size_factors = model.get_latent_library_size(a_val)
#
# ##
#
# if m and COMPLETE_RUN:
#     area_ds = AreaFilteredDataset("validation")
#
#     l = []
#     for x in tqdm(area_ds, desc="merging"):
#         l.append(x)
#     areas = np.concatenate(l, axis=0)
#
# ##
# if m and COMPLETE_RUN:
#     from scipy.stats import pearsonr
#
#     print(size_factors.shape)
#     print(areas.shape)
#     r, p = pearsonr(size_factors.ravel(), areas.ravel())
#     plt.figure()
#     plt.scatter(size_factors, areas, s=0.5)
#     plt.xlabel("latent size factors")
#     plt.ylabel("cell area")
#     plt.title(f"r: {r:0.2f} (p: {p:0.2f})")
#     plt.show()
#
# ##
# # imputation benchmark
#
#
# def get_corrupted_entries(split: str):
#     ds = PerturbedCellDataset(split)
#     ds.perturb()
#     corrupted_entries = ds.corrupted_entries.numpy()
#     # just a hash
#     h = np.sum(np.concatenate(np.where(corrupted_entries == 1)))
#     print(f"corrupted entries hash ({split}):", h)
#     return corrupted_entries
#
#
# if e_():
#     ce_train = get_corrupted_entries("train")
#     ce_val = get_corrupted_entries("validation")
#
# ##
# if e_():
#     ds = SumFilteredDataset("train")
#
#     @memory.cache
#     def f_ncqlliwr2(ds):
#         l0 = []
#         for i, x in enumerate(tqdm(ds, "merging")):
#             l0.append(x)
#         return l0
#
#     l0 = f_ncqlliwr2(ds)
#     raw = np.concatenate(l0, axis=0)
#     raw[ce_train] = 0
#     raw = np.round(raw)
#     raw = raw.astype(int)
#     a_perturbed = ad.AnnData(raw)
#
# ##
# if e_():
#     scvi.data.setup_anndata(a_perturbed)
#     # TRAIN_PERTURBED = True
#     TRAIN_PERTURBED = False
#     if TRAIN_PERTURBED:
#         # to navigate there with PyCharm and set a breakpoint on a warning (haven't done yet)
#         import scvi.core.distributions
#
#         model = scvi.model.SCVI(a_perturbed)
#     if TRAIN_PERTURBED:
#         model.train(
#             train_size=1.0, n_epochs=N_EPOCHS, n_epochs_kl_warmup=N_EPOCHS_KL_WARMUP
#         )
#         f = file_path("scvi_model_perturbed.scvi")
#         if os.path.isdir(f):
#             shutil.rmtree(f)
#         model.save(f)
#     else:
#         model = scvi.model.SCVI.load(file_path("scvi_model_perturbed.scvi"), adata=a)
#     print(model.get_elbo())
#
# ##
# if e_():
#     x_val_perturbed = a_val.X.copy()
#     x_val_perturbed[ce_val] = 0
#     a_val_perturbed = ad.AnnData(x_val_perturbed)
#
# ##
# if e_():
#     p = model.get_likelihood_parameters(a_val_perturbed)
#     from scvi.core.distributions import ZeroInflatedNegativeBinomial
#
#     x_val_perturbed_pred = ZeroInflatedNegativeBinomial(
#         mu=torch.tensor(p["mean"]),
#         theta=torch.tensor(p["dispersions"]),
#         zi_logits=torch.tensor(p["dropout"]),
#     ).mean.numpy()
#
# ##
# if e_():
#     # ne: normal entries
#     ne_train = np.logical_not(ce_train)
#     ne_val = np.logical_not(ce_val)
#     x_val = a_val.X.copy()
#
#     uu0 = x_val_perturbed_pred[ce_val]
#     uu1 = x_val[ce_val]
#
#     vv0 = x_val_perturbed_pred[ne_val]
#     vv1 = x_val[ne_val]
# ##
# if e_():
#     fig = plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.hist(np.abs(uu0 - uu1))
#     m = np.mean(np.abs(uu0 - uu1))
#     plt.title(f"scores for imputed entries\nmean: {m:0.2f}")
#     plt.yscale("log")
#
#     plt.subplot(1, 2, 2)
#     plt.hist(np.abs(vv0 - vv1))
#     m = np.mean(np.abs(vv0 - vv1))
#     plt.title(f"control: normal entries\nmean: {m:0.2f}")
#     plt.yscale("log")
#
#     fig.suptitle("abs(original vs predicted)")
#     plt.tight_layout()
#     plt.show()
#
# ##
# if e_():
#     from analyses.aa_reconstruction_benchmark.aa_ad_reconstruction import (
#         Prediction,
#         Space,
#     )
#
#     s = np.abs(uu0 - uu1)
#     t = np.abs(vv0 - vv1)
#     Prediction.welch_t_test(s, t)
#     # the printed p-value is very close to 0
#     # conclusion: the score for imputed data is worse than the one from non-perturbed data; this is expected and the
#     # alternative case would have been a model whose scores are both bad because it is not properly trained
# ##
#
# if e_():
#     kwargs = dict(
#         original=x_val,
#         corrupted_entries=ce_val,
#         predictions_from_perturbed=x_val_perturbed_pred,
#         space=Space.raw_sum.value,
#         name="scVI",
#         split="validation",
#     )
#     scvi_predictions = Prediction(**kwargs)
#
#     scvi_predictions.plot_reconstruction()
#     # scvi_predictions.plot_scores()
#
# ##
# if m and False:
#     p = scvi_predictions.transform_to(Space.scaled_mean)
#     p.name = "scVI scaled"
#     p.plot_reconstruction()
#     # p.plot_scores()
#
# ##
# import pickle
# import dill
#
# if e_():
#     d = {"scVI": kwargs}
#     dill.dump(d, open(file_path("scvi_scores.pickle"), "wb"))
#
# ##
# if e_():
#     pickle.dump(
#         {"input": aa_val, "latent": bb_val},
#         open(file_path("latent_anndata_from_scvi.pickle"), "wb"),
#     )
