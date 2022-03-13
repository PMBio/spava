import os



def test_imc_data_scvi():
    os.environ["SPATIALMUON_TEST"] = "analyses/scvi_analyses/imc_data_scvi.py"
    import analyses.scvi_analyses.imc_data_scvi


if __name__ == "__main__":
    test_imc_data_scvi()
