import importlib
import os
from tests.testing_utils import is_debug


def test_preprocess_visium_endometrium():
    os.environ["SPATIALMUON_TEST"] = "datasets/visium_endometrium.py"
    import datasets.visium_endometrium


def test_preprocess_visium_endometrium_tiler32():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/visium_endometrium_tiler.py"
    os.environ["SPATIALMUON_FLAGS"] = "TILE_SIZE=32"
    import datasets.tilers.visium_endometrium_tiler

    del os.environ["SPATIALMUON_FLAGS"]


def test_preprocess_visium_endometrium_tiler_large():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/visium_endometrium_tiler.py"
    os.environ["SPATIALMUON_FLAGS"] = "TILE_SIZE=large"
    import datasets.tilers.visium_endometrium_tiler

    importlib.reload(datasets.tilers.visium_endometrium_tiler)
    del os.environ["SPATIALMUON_FLAGS"]


# def test_preprocess_visium_endometrium_graphs():
#     os.environ["SPATIALMUON_TEST"] = "datasets/visium_endometrium_graphs.py"
#     import datasets.graphs.visium_endometrium_graphs


def test_visium_endometrium_loaders():
    os.environ["SPATIALMUON_TEST"] = "datasets/loaders/visium_endometrium_loaders.py"
    import datasets.loaders.visium_endometrium_loaders

# -----------------

def test_train_expression_vae():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium/expression_vae_runner.py"
    os.environ["SPATIALMUON_FLAGS"] = "DATASET_NAME=visium_endometrium"
    import analyses.visium.expression_vae_runner

    del os.environ['SPATIALMUON_FLAGS']

def test_analyze_expression_vae():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium_endometrium/expression_vae_analysis.py"
    os.environ["SPATIALMUON_FLAGS"] = "MODEL_NAME=expression_vae"
    import analyses.visium_endometrium.expression_vae_analysis

    del os.environ["SPATIALMUON_FLAGS"]

# -----------------

def test_train_image_expression_conv_vae32():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium/image_expression_conv_vae_runner.py"
    os.environ["SPATIALMUON_FLAGS"] = "TILE_SIZE=32,DATASET_NAME=visium_endometrium"

    import analyses.visium.image_expression_conv_vae_runner

    del os.environ["SPATIALMUON_FLAGS"]


def test_analyze_image_expression_conv_vae32():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium_endometrium/expression_vae_analysis.py"
    os.environ[
        "SPATIALMUON_FLAGS"
    ] = "MODEL_NAME=image_expression_conv_vae,TILE_SIZE=32"
    import analyses.visium_endometrium.expression_vae_analysis

    importlib.reload(analyses.visium_endometrium.expression_vae_analysis)

    del os.environ["SPATIALMUON_FLAGS"]

# -----------------

def test_train_image_expression_conv_vae_large():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium/image_expression_conv_vae_runner.py"
    os.environ["SPATIALMUON_FLAGS"] = "TILE_SIZE=large,DATASET_NAME=visium_endometrium"

    import analyses.visium.image_expression_conv_vae_runner

    importlib.reload(analyses.visium.image_expression_conv_vae_runner)

    del os.environ["SPATIALMUON_FLAGS"]


def test_analyze_image_expression_conv_vae_large():
    os.environ[
        "SPATIALMUON_TEST"
    ] = "analyses/visium_endometrium/expression_vae_analysis.py"
    os.environ[
        "SPATIALMUON_FLAGS"
    ] = "MODEL_NAME=image_expression_conv_vae,TILE_SIZE=large"
    import analyses.visium_endometrium.expression_vae_analysis

    importlib.reload(analyses.visium_endometrium.expression_vae_analysis)

    del os.environ["SPATIALMUON_FLAGS"]

# -----------------

if __name__ == "__main__":
    # if is_debug():
    pass
