import os


def test_preprocess_imc_data():
    os.environ["SPATIALMUON_TEST"] = "datasets/imc_data.py"
    import datasets.imc_data


def test_preprocess_imc_jeongbin_data():
    os.environ["SPATIALMUON_TEST"] = "datasets/imc_jeongbin_data.py"
    import datasets.imc_jeongbin_data


def test_preprocess_visium_data():
    os.environ["SPATIALMUON_TEST"] = "datasets/visium_data.py"
    import datasets.visium_data


if __name__ == "__main__":
    test_preprocess_imc_data()
    test_preprocess_imc_jeongbin_data()
    test_preprocess_visium_data()
