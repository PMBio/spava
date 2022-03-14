import os
from tests.testing_utils import is_debug


def test_preprocess_imc_data():
    os.environ["SPATIALMUON_TEST"] = "datasets/imc_data.py"
    import datasets.imc_data


def test_preprocess_imc_data_tiler():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/imc_data_tiler.py"
    import datasets.tilers.imc_data_tiler


def test_preprocess_imc_data_graphs():
    os.environ["SPATIALMUON_TEST"] = "datasets/graphs/imc_data_graphs.py"
    import datasets.graphs.imc_data_graphs


def test_imc_data_loaders():
    os.environ["SPATIALMUON_TEST"] = "datasets/loaders/imc_data_loaders.py"
    import datasets.loaders.imc_data_loaders


def test_train_vae_expression():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/expression_vae_runner.py"
    import analyses.imc.expression_vae_runner


def test_analyze_expression_model():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/expression_vae_analysis.py"
    import analyses.imc.expression_vae_analysis


if __name__ == "__main__":
    if is_debug():
        pass
