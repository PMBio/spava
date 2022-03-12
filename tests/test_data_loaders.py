import os


def test_imc_data_loaders():
    os.environ["SPATIALMUON_TEST"] = "datasets/loaders/imc_data_loaders.py"
    import datasets.loaders.imc_data_loaders


if __name__ == "__main__":
    test_imc_data_loaders()
