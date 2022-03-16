import importlib
import os
from tests.testing_utils import is_debug


def test_preprocess_visium_mousebrain():
    os.environ["SPATIALMUON_TEST"] = "datasets/visium_mousebrain.py"
    import datasets.visium_mousebrain


def test_preprocess_visium_mousebrain_tiler():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/visium_mousebrain_tiler.py"
    import datasets.tilers.visium_mousebrain_tiler


def test_preprocess_visium_mousebrain_graphs():
    os.environ["SPATIALMUON_TEST"] = "datasets/visium_mousebrain_graphs.py"
    import datasets.graphs.visium_mousebrain_graphs


def test_visium_mousebrain_loaders():
    os.environ["SPATIALMUON_TEST"] = "datasets/loaders/visium_mousebrain_loaders.py"
    import datasets.loaders.visium_mousebrain_loaders


def test_train_expression_vae():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium_mousebrain/expression_vae_runner.py"
    import analyses.visium_mousebrain.expression_vae_runner


def test_analyze_expression_vae():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium_mousebrain/expression_vae_analysis.py"
    os.environ["SPATIALMUON_FLAGS"] = "MODEL_NAME=expression_vae"
    import analyses.visium_mousebrain.expression_vae_analysis

    del os.environ["SPATIALMUON_FLAGS"]


def test_train_image_expression_conv_vae():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium_mousebrain/image_expression_conv_vae_runner.py"
    import analyses.visium_mousebrain.image_expression_conv_vae_runner


def test_analyze_image_expression_conv_vae():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium_mousebrain/expression_vae_analysis.py"
    os.environ["SPATIALMUON_FLAGS"] = "MODEL_NAME=image_expression_conv_vae"
    import analyses.visium_mousebrain.expression_vae_analysis

    importlib.reload(analyses.visium_mousebrain.expression_vae_analysis)

    del os.environ["SPATIALMUON_FLAGS"]


if __name__ == "__main__":
    # if is_debug():
    test_visium_mousebrain_loaders()
