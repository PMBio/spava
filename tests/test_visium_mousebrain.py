import os
from tests.testing_utils import is_debug


def test_preprocess_visium_data():
    os.environ["SPATIALMUON_TEST"] = "datasets/visium_data.py"
    import datasets.visium_data


def test_preprocess_visium_data_tiler():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/visium_data_tiler.py"
    import datasets.tilers.visium_data_tiler


def test_preprocess_visium_data_graphs():
    os.environ["SPATIALMUON_TEST"] = "datasets/visium_data_graphs.py"
    import datasets.graphs.visium_data_graphs


def test_train_vae_expression():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium_mousebrain/expression_vae_runner.py"
    import analyses.visium_mousebrain.expression_vae_runner


def test_analyze_expression_model():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium_mousebrain/expression_vae_analysis.py"
    import analyses.visium_mousebrain.expression_vae_analysis


if __name__ == "__main__":
    if is_debug():
        pass
