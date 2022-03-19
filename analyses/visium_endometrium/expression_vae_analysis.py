##

import os
import shutil
import tempfile

import colorama
import anndata as ad
import matplotlib.pyplot as plt
import optuna
import spatialmuon
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from analyses.analisys_utils import louvain_plot, scanpy_compute
from analyses.imputation_score import Prediction
from datasets.loaders.visium_endometrium_loaders import get_cells_data_loader
from utils import get_execute_function, parse_flags

e_ = get_execute_function()

##
flags = parse_flags(default={"MODEL_NAME": "expression_vae"})
MODEL_NAME = flags["MODEL_NAME"]
is_expression_vae = MODEL_NAME == "expression_vae"
is_image_expression_conv_vae = MODEL_NAME == "image_expression_conv_vae"

##
if is_expression_vae:
    MODEL_FULLNAME = f'visium_endometrium_{MODEL_NAME}'
elif is_image_expression_conv_vae:
    TILE_SIZE = flags['TILE_SIZE']
    MODEL_FULLNAME = f'visium_endometrium_{MODEL_NAME}_{TILE_SIZE}'
else:
    assert False

##
if is_expression_vae:
    from models.expression_vae import VAE
elif is_image_expression_conv_vae:
    from models.image_expression_conv_vae import VAE
else:
    assert False

##
if e_():
    from utils import file_path

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    study = optuna.load_study(
        study_name=MODEL_FULLNAME,
        storage="sqlite:///" + file_path(f"optuna_{MODEL_FULLNAME}.sqlite"),
    )
    print("best trial:")
    print(study.best_trial)
    print(study.best_trial.user_attrs["version"])

    # re-train the best model but by perturbing the dataset
    if False:
        # if True:
        if False:
            # if True:
            from analyses.visium.expression_vae_runner import objective

            trial = study.best_trial
            trial.set_user_attr("MAX_EPOCHS", 50)
            objective(trial)
            sys.exit(0)
        else:
            # manually update version from the just trained perturbed model
            if is_expression_vae:
                pass
            elif is_image_expression_conv_vae:
                pass
            else:
                assert False
    else:
        version = study.best_trial.user_attrs["version"]

##
if e_():
    MODEL_CHECKPOINT = file_path(
        f"checkpoints/{MODEL_FULLNAME}/version_{version}/checkpoints/last.ckpt"
    )
    expression_vae = VAE.load_from_checkpoint(MODEL_CHECKPOINT)


def get_latent_representation(loader, model):
    print("merging expressions and computing embeddings... ", end="")
    all_mu = []
    all_expression = []
    all_a = []
    all_b = []
    all_is_corrupted = []
    for data in tqdm(loader, desc="embedding expression"):
        if is_expression_vae:
            expression, is_corrupted = data
            a, b, mu, std, z = model(expression)
        elif is_image_expression_conv_vae:
            image_input, expression, is_corrupted = model.unfold_batch(data)
            a, b, mu, std, z = model(expression, image_input)
        else:
            assert False
        all_mu.append(mu.detach())
        all_expression.append(expression)
        all_a.append(a.detach())
        all_b.append(b.detach())
        all_is_corrupted.append(is_corrupted)
    mus = torch.cat(all_mu, dim=0)
    expressions = torch.cat(all_expression, dim=0)
    aas = torch.cat(all_a, dim=0)
    bbs = torch.cat(all_b, dim=0)
    are_perturbed = torch.cat(all_is_corrupted, dim=0)

    mus = ad.AnnData(mus.detach().numpy())
    expressions = ad.AnnData(expressions.detach().numpy())

    reconstructed = model.expected_value(aas, bbs)
    return mus, expressions, reconstructed, are_perturbed


if e_():
    if is_expression_vae:
        batch_size = 1023
    elif is_image_expression_conv_vae:
        batch_size = 127
    else:
        assert False

    if is_expression_vae:
        only_expression = True
    elif is_image_expression_conv_vae:
        only_expression = False
    else:
        assert False

    nw = 10
    # nw = 0
    print(f'{colorama.Fore.MAGENTA}nw = {nw}{colorama.Fore.RESET}')

    train_loader_non_perturbed = get_cells_data_loader(
        split="train", batch_size=batch_size, only_expression=only_expression, num_workers=nw
    )

    val_loader_non_perturbed = get_cells_data_loader(
        split="validation", batch_size=batch_size, only_expression=only_expression, num_workers=nw
    )
    val_loader_perturbed = get_cells_data_loader(
        split="validation", batch_size=batch_size, perturb=True, only_expression=only_expression, num_workers=nw
    )

    test_loader_non_perturbed = get_cells_data_loader(
        split="test", batch_size=batch_size, only_expression=only_expression, num_workers=nw
    )

    (
        mus_train_non_perturbed,
        expression_train_non_perturbed,
        _,
        _,
    ) = get_latent_representation(
        loader=train_loader_non_perturbed,
        model=expression_vae,
    )

    (
        mus_val_non_perturbed,
        expression_val_non_perturbed,
        _,
        _,
    ) = get_latent_representation(
        loader=val_loader_non_perturbed,
        model=expression_vae,
    )
    (
        mus_val_perturbed,
        expression_val_perturbed,
        val_perturbed_reconstructed,
        val_are_perturbed,
    ) = get_latent_representation(
        loader=val_loader_perturbed,
        model=expression_vae,
    )

    (
        mus_test_non_perturbed,
        expression_test_non_perturbed,
        _,
        _,
    ) = get_latent_representation(
        loader=test_loader_non_perturbed,
        model=expression_vae,
    )

##
if e_():
    scanpy_compute(expression_val_non_perturbed)
    scanpy_compute(mus_val_non_perturbed)

    try:
        louvain_plot(expression_val_non_perturbed, f"expression val (non perturbed)\n{MODEL_FULLNAME}")
    except KeyError as e:
        if "SPATIALMUON_TEST" in os.environ and str(e) == "'X_umap'":
            print("not plotting UMAP for edge case in test")
        else:
            raise e

    try:
        louvain_plot(mus_val_non_perturbed, f"latent val (non perturbed)\n{MODEL_FULLNAME}")
    except KeyError as e:
        if "SPATIALMUON_TEST" in os.environ and str(e) == "'X_umap'":
            print("not plotting UMAP for edge case in test")
        else:
            raise e

##
if e_():
    merged = ad.AnnData.concatenate(
        mus_train_non_perturbed,
        mus_val_non_perturbed,
        mus_test_non_perturbed,
        batch_categories=["train", "validation", "test"],
    )
    scanpy_compute(merged)
    lou = merged.obs["louvain"]

    from datasets.visium_endometrium import get_split_bimap, get_smu_file

    train_map_left, _ = get_split_bimap('train')
    val_map_left, _ = get_split_bimap('validation')
    test_map_left, _ = get_split_bimap('test')

    louvain_train = lou.iloc[: len(train_map_left)]
    louvain_val = lou.iloc[len(train_map_left) : len(lou) - len(test_map_left)]
    louvain_test = lou.iloc[len(lou) - len(test_map_left) :]

    categories = lou.cat.categories.tolist() + ["Nein"]

    assert len(louvain_train) + len(louvain_val) + len(louvain_test) == len(lou)
    assert all([s.endswith("-train") for s in louvain_train.index.tolist()])
    assert all([s.endswith("-validation") for s in louvain_val.index.tolist()])
    assert all([s.endswith("-test") for s in louvain_test.index.tolist()])

    from datasets.visium_endometrium import visium_endrometrium_samples
    for sample in visium_endrometrium_samples:
        louvain_train_indices = np.array([k for k, (kk, _) in train_map_left.items() if kk == sample])
        louvain_val_indices = np.array([k for k, (kk, _) in val_map_left.items() if kk == sample])
        louvain_test_indices = np.array([k for k, (kk, _) in test_map_left.items() if kk == sample])

        train_indices = np.array([vv for kk, vv in train_map_left.values() if kk == sample])
        val_indices = np.array([vv for kk, vv in val_map_left.values() if kk == sample])
        test_indices = np.array([vv for kk, vv in test_map_left.values() if kk == sample])
        n = len(train_indices) + len(val_indices) + len(test_indices)

        ordered_lou = pd.Categorical(["Nein"] * n, categories=categories)

        ordered_lou[train_indices] = louvain_train[louvain_train_indices].to_numpy()
        ordered_lou[val_indices] = louvain_val[louvain_val_indices].to_numpy()
        ordered_lou[test_indices] = louvain_test[louvain_test_indices].to_numpy()

        assert ordered_lou.value_counts()["Nein"] == 0
        ordered_lou = ordered_lou.remove_categories("Nein")
        lou_for_smu = ordered_lou.astype("category")

        # .clone() does not currently work in read_only=True, so let's go for a workaround
        with tempfile.TemporaryDirectory() as td:
            des = os.path.join(td, 'temp.h5smu')
            s = get_smu_file(sample=sample, read_only=True)
            src = s.backing.filename
            s.backing.close()
            shutil.copyfile(src, des)
            s = spatialmuon.SpatialMuData(des)
            s["visium"]["processed"].obs["vae"] = lou_for_smu
            s["visium"]["processed"].masks.obj_has_changed("obs")

        ##
            _, ax = plt.subplots(1, figsize=(15, 15))
            s["visium"]["image"].plot(ax=ax)
            s["visium"]["processed"].masks.plot("vae", ax=ax)
            plt.title(f"latent space from VAE model\n{MODEL_FULLNAME}")
            plt.show()

##
if e_():
    # if True:
    kwargs = dict(
        original=expression_val_non_perturbed.X,
        corrupted_entries=val_are_perturbed.detach().cpu().numpy(),
        predictions_from_perturbed=val_perturbed_reconstructed.detach().cpu().numpy(),
        name=f"{MODEL_FULLNAME} (val)",
    )
    vae_predictions = Prediction(**kwargs)

    plt.style.use("default")
    # vae_predictions.plot_reconstruction()
    vae_predictions.plot_scores(hist=True)
    vae_predictions.plot_summary()


# ##
# if e_():
#     import scanpy as sc
#     adata = expression_train_non_perturbed
#     sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
#     sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)
#     print(f'np.mean(adata.X, axis=0) = {np.mean(adata.X, axis=0)}')
#     print(f'np.std(adata.X, axis=0) = {np.std(adata.X, axis=0)}')

##
import pickle

#
if e_():
    f = file_path("visium_endometrium/imputation_scores")
    os.makedirs(f, exist_ok=True)

    d = {"vanilla VAE": kwargs}
    pickle.dump(d, open(file_path(f"{f}/{MODEL_FULLNAME}.pickle"), "wb"))
