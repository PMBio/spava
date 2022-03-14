import os
from tests.testing_utils import is_debug


def test_preprocess_imc():
    os.environ["SPATIALMUON_TEST"] = "datasets/imc.py"
    import datasets.imc


def test_preprocess_imc_tiler():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/imc_tiler.py"
    import datasets.tilers.imc_tiler


def test_preprocess_imc_graphs():
    os.environ["SPATIALMUON_TEST"] = "datasets/graphs/imc_graphs.py"
    import datasets.graphs.imc_graphs


def test_imc_loaders():
    os.environ["SPATIALMUON_TEST"] = "datasets/loaders/imc_loaders.py"
    import datasets.loaders.imc_loaders


def test_train_vae_expression():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/expression_vae_runner.py"
    import analyses.imc.expression_vae_runner


def test_analyze_expression_model():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/expression_vae_analysis.py"
    import analyses.imc.expression_vae_analysis


if __name__ == "__main__":
    if is_debug():
        pass
