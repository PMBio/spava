import os
from tests.testing_utils import is_debug


def test_preprocess_imc_data():
    os.environ["SPATIALMUON_TEST"] = "datasets/imc_data.py"
    import datasets.imc_data


def test_preprocess_imc_data_graphs():
    os.environ["SPATIALMUON_TEST"] = "datasets/graphs/imc_data_graphs.py"
    import datasets.graphs.imc_data_graphs


def test_preprocess_imc_data_tiler():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/imc_data_tiler.py"
    import datasets.tilers.imc_data_tiler


# --
def test_preprocess_visium_data():
    os.environ["SPATIALMUON_TEST"] = "datasets/visium_data.py"
    import datasets.visium_data


def test_preprocess_visium_data_graphs():
    os.environ["SPATIALMUON_TEST"] = "datasets/visium_data_graphs.py"
    import datasets.graphs.visium_data_graphs


def test_preprocess_visium_data_tiler():
    os.environ["SPATIALMUON_TEST"] = "datasets/tilers/visium_data_tiler.py"
    import datasets.tilers.visium_data_tiler


# --
def test_preprocess_imc_jeongbin_data():
    os.environ["SPATIALMUON_TEST"] = "datasets/imc_jeongbin_data.py"
    import datasets.imc_jeongbin_data


if __name__ == "__main__":
    if not is_debug():
        test_preprocess_imc_data()
        test_preprocess_imc_data_graphs()
        test_preprocess_imc_data_tiler()

        test_preprocess_visium_data()
        test_preprocess_visium_data_graphs()
        test_preprocess_visium_data_tiler()

        test_preprocess_imc_jeongbin_data()
    else:
        test_preprocess_visium_data()