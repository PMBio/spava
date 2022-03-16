import os
import importlib
from tests.testing_utils import is_debug


def test_preprocess_imc():
    os.environ["SPATIALMUON_TEST"] = "datasets/imc.py"
    import datasets.imc


def test_preprocess_imc_tiler():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/imc_tiler.py"
    import datasets.tilers.imc_tiler


def test_preprocess_imc_pca_tiler():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/imc_pca_tiler.py"
    import datasets.tilers.imc_pca_tiler


def test_preprocess_imc_graphs():
    os.environ["SPATIALMUON_TEST"] = "datasets/graphs/imc_graphs.py"
    import datasets.graphs.imc_graphs


def test_imc_loaders():
    os.environ["SPATIALMUON_TEST"] = "datasets/loaders/imc_loaders.py"
    import datasets.loaders.imc_loaders


def test_train_expression_vae():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/expression_vae_runner.py"
    import analyses.imc.expression_vae_runner


def test_analyze_expression_vae():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/expression_vae_analysis.py"
    os.environ["SPATIALMUON_FLAGS"] = "expression_vae"
    import analyses.imc.expression_vae_analysis

    del os.environ["SPATIALMUON_FLAGS"]


def test_train_image_expression_vae():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/image_expression_vae_runner.py"
    import analyses.imc.image_expression_vae_runner


def test_analyze_image_expression_vae():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/expression_vae_analysis.py"
    os.environ["SPATIALMUON_FLAGS"] = "image_expression_vae"
    import analyses.imc.expression_vae_analysis

    importlib.reload(analyses.imc.expression_vae_analysis)

    del os.environ["SPATIALMUON_FLAGS"]


def test_train_image_expression_conv_vae():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/image_expression_conv_vae_runner.py"
    import analyses.imc.image_expression_conv_vae_runner


def test_analyze_image_expression_conv_vae():
    os.environ["SPATIALMUON_TEST"] = "analyses/imc/expression_conv_vae_analysis.py"
    os.environ["SPATIALMUON_FLAGS"] = "image_expression_conv_vae"
    import analyses.imc.expression_vae_analysis

    importlib.reload(analyses.imc.expression_vae_analysis)

    del os.environ["SPATIALMUON_FLAGS"]


if __name__ == "__main__":
    # test_preprocess_imc_tiler()
    # test_train_image_expression_conv_vae()
    test_analyze_image_expression_conv_vae()
    # test_imc_loaders()
# if is_debug():
# pass
# test_analyze_expression_vae()
# test_analyze_image_expression_vae()
