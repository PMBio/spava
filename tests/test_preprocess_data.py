import os

os.environ["SPATIALMUON_TEST"] = "aaa"


def test_preprocess_imc_data():
    import datasets.imc_data


def test_preprocess_imc_jeongbin_data():
    import datasets.imc_jeongbin_data


def test_preprocess_visium_data():
    import datasets.visium_data


if __name__ == "__main__":
    test_preprocess_imc_data()
    test_preprocess_imc_jeongbin_data()
    test_preprocess_visium_data()
