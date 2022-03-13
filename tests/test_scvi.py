import os


def test_imc_data_scvi():
    os.environ["SPATIALMUON_TEST"] = "analyses/scvi_analyses/imc_data_scvi.py"
    import analyses.scvi_analyses.imc_data_scvi


def test_visium_data_scvi():
    os.environ["SPATIALMUON_TEST"] = "analyses/scvi_analyses/visium_data_scvi.py"
    import analyses.scvi_analyses.visium_data_scvi


if __name__ == "__main__":
    test_imc_data_scvi()
    test_visium_data_scvi()
