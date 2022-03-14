##
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm

from analyses.essentials import *
from old_code.data2 import PerturbedRGBCells
from old_code.graphs import CellExpressionGraphOptimized

m = __name__ == "__main__"

PLOT = True
BUG = False

#
if m:
    SPLIT = "validation"
    # MODEL = 'conv_encoder'
    # MODEL = "resnet_encoder"
    MODEL = "ai_gnn_vae"
    assert MODEL in ["conv_encoder", "resnet_encoder", "ai_gnn_vae"]
    from old_code.models import VAE as ImageToExpressionVAE
    from old_code.models import ResNetToExpression
    from old_code.models import GnnVae

    if MODEL == "conv_encoder":
        model_index_non_perturbed = 116
        model_index_perturbed = 151
        tensorboard_label = "image_to_expression"
        model = ImageToExpressionVAE
        ds_original = PerturbedRGBCells(split=SPLIT)
        ds_perturbed = PerturbedRGBCells(split=SPLIT)
        ds_perturbed.perturb()
        data_loader_class = DataLoader
    elif MODEL == "resnet_encoder":
        model_index_non_perturbed = 154
        model_index_perturbed = 162
        model = ResNetToExpression
        tensorboard_label = "image_to_expression"
        ds_original = PerturbedRGBCells(split=SPLIT)
        ds_perturbed = PerturbedRGBCells(split=SPLIT)
        ds_perturbed.perturb()
        data_loader_class = DataLoader
    elif MODEL == "ai_gnn_vae":
        model_index_non_perturbed = 122
        model = GnnVae
        tensorboard_label = "gnn_vae"
        ds_original = CellExpressionGraphOptimized(split=SPLIT, graph_method="gaussian")
        # PERTURB_ENTIRE_CELLS = False
        PERTURB_ENTIRE_CELLS = True
        if PERTURB_ENTIRE_CELLS:
            ds_perturbed = CellExpressionGraphOptimized(
                split=SPLIT, graph_method="gaussian", perturb_entire_cells=True
            )
            model_index_perturbed = 122
        else:
            ds_perturbed = CellExpressionGraphOptimized(
                split=SPLIT, graph_method="gaussian", perturb=True
            )
            model_index_perturbed = 128
        data_loader_class = GeometricDataLoader
    else:
        raise RuntimeError()

    f_original = (
        f"/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/{tensorboard_label}/version_"
        f"{model_index_non_perturbed}"
        "/checkpoints/last.ckpt"
    )
    model_original = model.load_from_checkpoint(f_original)
    model_original.cuda()
    loader_original = data_loader_class(
        ds_original, batch_size=1024, num_workers=8, pin_memory=True
    )

    f_perturbed = (
        f"/data/l989o/deployed/a/data/spatial_uzh_processed/a/checkpoints/{tensorboard_label}/version_"
        f"{model_index_perturbed}"
        "/checkpoints/last.ckpt"
    )
    model_perturbed = model.load_from_checkpoint(f_perturbed)
    model_perturbed.cuda()
    loader_perturbed = data_loader_class(
        ds_perturbed, batch_size=1024, num_workers=8, pin_memory=True
    )


#


def get_list_of_z(loader, model):
    list_of_z = []
    with torch.no_grad():
        for data in tqdm(loader_original, desc="forwarding"):
            if MODEL == "ai_gnn_vae":
                data.to(model_original.device)
                output = model_original(
                    data.x, data.edge_index, data.edge_attr, data.is_center
                )
                z = [zz.cpu() for zz in output]
            else:
                data = [d.to(model_original.device) for d in data]
                _, x, mask, _ = data
                z = [zz.cpu() for zz in model_original(x, mask)]
            list_of_z.append(z)
    torch.cuda.empty_cache()
    return list_of_z


if m and PLOT:
    list_of_z = get_list_of_z(loader=loader_original, model=model_original)

    #
    l = []
    for zz in list_of_z:
        a, mu, std, z = zz
        l.append(mu)
        # reconstructed = model_original.get_dist(a).mean
    mus = torch.cat(l, dim=0).numpy()
    #
    from analyses.essentials import scanpy_compute
    import scanpy as sc
    import anndata as ad

    a = ad.AnnData(mus)
    sc.tl.pca(a)
    sc.pl.pca(a)
    #
    from utils import reproducible_random_choice

    random_indices = reproducible_random_choice(len(a), 10000)

    #
    b = a[random_indices]

    #
    scanpy_compute(b)

    #
    sc.pl.umap(b, color="louvain", title=f"{MODEL} latent space")
    import pickle

    pickle.dump(
        {"latent": b}, open(file_path(f"latent_anndata_from_{MODEL}.pickle"), "wb")
    )
##
if m:
    list_of_z_perturbed = get_list_of_z(loader=loader_perturbed, model=model_perturbed)

##

if m:
    l = []
    for zz in list_of_z_perturbed:
        a, mu, std, z = zz
        xx = model_original.get_dist(a).mean
        l.append(xx)
    reconstructed = torch.cat(l, dim=0).numpy()

##
if m:
    l = []
    for data in tqdm(loader_original, desc="merging expression"):
        if MODEL == "ai_gnn_vae":
            expression = data.x[torch.where(data.is_center == 1.0)[0], :]
        else:
            expression, _, _, _ = data
        l.append(expression)
    expressions = torch.cat(l, dim=0).numpy()

    l = []
    for data in tqdm(loader_perturbed, desc="merging perturbed entries"):
        if MODEL == "ai_gnn_vae":
            is_perturbed = data.is_perturbed[torch.where(data.is_center == 1.0)[0], :]
        else:
            _, _, _, is_perturbed = data
        l.append(is_perturbed)
    are_perturbed = torch.cat(l, dim=0).numpy()

##
if m:
    h = np.sum(np.concatenate(np.where(are_perturbed == 1)))
    print("corrupted entries hash:", h)

##
if m and PLOT:
    kwargs = dict(
        original=expressions,
        corrupted_entries=are_perturbed,
        predictions_from_perturbed=reconstructed,
        space=Space.scaled_mean,
        name=f"{MODEL} to expression",
        split="validation",
    )
    p = Prediction(**kwargs)
    p.plot_reconstruction()
    # p.plot_scores()

    p_raw = p.transform_to(Space.raw_sum)
    # p_raw.plot_reconstruction()
    # p_raw.plot_scores()

##
if m:
    p.plot_summary()

##
if m:
    import dill

    d = {f"{MODEL} expression": kwargs}
    dill.dump(d, open(file_path(f"{MODEL}_scores.pickle"), "wb"))

##
if m and BUG:
    cell_ds_original = ds_original.cell_expression_graph.cell_ds
    cell_ds_perturbed = ds_perturbed.cell_expression_graph.cell_ds
    ce = cell_ds_perturbed.corrupted_entries
    h = torch.sum(torch.cat(torch.where(ce == 1)))
    print(
        "corrupted entries hash:",
        h,
    )
    assert len(cell_ds_perturbed) == len(cell_ds_original)
    assert len(cell_ds_perturbed) == len(ds_original)
    assert len(cell_ds_perturbed) == len(ds_perturbed)
    wrong = []
    very_wrong = []
    for i in tqdm(range(len(cell_ds_perturbed))):

        def get_full_tensors(i):
            expression, _, _ = cell_ds_original[i]
            _, _, is_perturbed = cell_ds_perturbed[i]
            expression = torch.from_numpy(expression)
            expression_graph = ds_original[i].x
            is_perturbed_graph = ds_perturbed[i].is_perturbed
            is_center0 = ds_original[i].is_center
            is_center1 = ds_perturbed[i].is_center
            assert torch.equal(is_center0, is_center1)
            return (
                expression,
                expression_graph,
                is_perturbed,
                is_perturbed_graph,
                is_center0,
            )

        def get_tensors(i):
            (
                expression,
                expression_graph,
                is_perturbed,
                is_perturbed_graph,
                is_center0,
            ) = get_full_tensors(i)
            # print(expression.shape, is_perturbed.shape)
            # print(expression_graph.shape, is_perturbed_graph.shape)
            expression_center = expression_graph[
                torch.where(is_center0 == 1.0), :
            ].flatten()
            is_perturbed_center = is_perturbed_graph[
                torch.where(is_center0 == 1.0), :
            ].flatten()
            # print(expression_center.shape)
            # print(is_perturbed_center.shape)
            return expression, expression_center, is_perturbed, is_perturbed_center

        def get_bb(i):
            (
                expression,
                expression_center,
                is_perturbed,
                is_perturbed_center,
            ) = get_tensors(i)
            b0 = torch.allclose(expression, expression_center)
            b1 = torch.equal(is_perturbed, is_perturbed_center)
            return b0, b1

        b0, b1 = get_bb(i)
        if (b0 + b1) % 2 != 0:
            very_wrong.append(i)
        if not (b0 and b1):
            wrong.append(i)
    ##
    n = len(cell_ds_perturbed) / 5
    w = len(wrong)
    vw = len(very_wrong)
    print(vw / n, w / n)
    ##
    # print(f'vw = {very_wrong}')
    # print(f'w = {wrong}')
    if w + vw > 0:
        print("bug")
        for i in very_wrong:
            b0, b1 = get_bb(i)
            print(b0, b1)
        i = wrong[0]
        ##
        b0, b1 = get_bb(i)
        print(b0, b1)
        (
            expression,
            expression_graph,
            is_perturbed,
            is_perturbed_graph,
            is_center0,
        ) = get_full_tensors(i)
        expression, expression_center, is_perturbed, is_perturbed_center = get_tensors(
            i
        )
        print(is_perturbed)
        print(is_perturbed_center)
        ##
        print(expression)
        print(expression_center)
        torch.set_printoptions(profile="full")
        print(expression_graph)
        torch.set_printoptions(profile="default")
        print(torch.where(is_center0 == 1.0))
        print(expression_graph[torch.where(is_center0 == 1.0), :])

        ##
        from old_code.data2 import IndexInfo

        ii = IndexInfo("validation")
        print(ii.filtered_begins[:10])
        print(very_wrong)
        print(wrong[:10])

        ##
        from old_code.graphs import CellExpressionGraph

        cell_expression_ds = CellExpressionGraph("validation", "gaussian")
        cell_expression_ds.merge()
        get_ome = cell_expression_ds.cell_graph.get_ome_index_from_cell_index
        ##
        last_ok = wrong[0] - 1
        first_wrong = wrong[0]
        first_very_wrong = very_wrong[0]

        ##
        def analyse(cell_index):
            ##
            # cell_index = first_very_wrong  # remember to comment this
            print("cell_index =", cell_index)
            ome_index, local_cell_index = get_ome(cell_index)
            print(f"ome_index = {ome_index}")
            data = cell_expression_ds[cell_index]
            from old_code.data2 import (
                ExpressionFilteredDataset,
                quantiles_for_normalization,
            )

            expression_filtered_dataset = ExpressionFilteredDataset(split="validation")
            expression_for_ome = expression_filtered_dataset[ome_index]
            expression_for_ome /= quantiles_for_normalization
            computed_local_cell_index = cell_index - ii.filtered_begins[ome_index]
            assert local_cell_index == computed_local_cell_index
            real_expression0 = expression_for_ome[computed_local_cell_index]
            real_expression1, _, _ = cell_ds_original[cell_index]
            assert np.allclose(real_expression0, real_expression1)
            if False:
                print("expression from ome_index:")
                print(real_expression0)
                print("expression from cell_ds_original")
                print(real_expression1)
                print("all expressions from ome_index:")
                print(expression_for_ome.shape)
                print(np.where((expression_for_ome == real_expression0).all(axis=1)))
                print(f"local_cell_index = {local_cell_index}")
                print(
                    np.where(
                        np.isclose(expression_for_ome, real_expression1).all(axis=1)
                    )
                )
                print(f"len(expression_for_ome) = {len(expression_for_ome)}")
            expression = data.x[data.center_index].numpy()
            assert np.allclose(expression, real_expression0)
            expression
            np.where(np.isclose(expression_for_ome, expression).all(axis=1))
            np.where(
                np.isclose(expression_filtered_dataset[ome_index - 1], expression).all(
                    axis=1
                )
            )
            np.where(
                np.isclose(expression_filtered_dataset[ome_index + 1], expression).all(
                    axis=1
                )
            )
            all_expressions = cell_expression_ds.merged_expressions.numpy()
            np.where(np.isclose(all_expressions, expression).all(axis=1))
            print(f"cell_index = {cell_index}")
            print(f"ome_index = {ome_index}")
            print(f"get_ome(1417) = {get_ome(1417)}")
            print(f"ii.filtered_begins[1] = {ii.filtered_begins[1]}")
            print(f"local_cell_index = {local_cell_index}")

        ##
        analyse(last_ok)
        analyse(first_wrong)
        analyse(first_very_wrong)
        ##
        # quick test
        cell_expression_ds_unmerged = CellExpressionGraph("validation", "gaussian")
        ##
        data = cell_expression_ds_unmerged[first_wrong]
        data.x[data.center_index]
        cell_ds_original[first_wrong][0]
