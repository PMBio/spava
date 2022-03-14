import os
from tests.testing_utils import is_debug


def test_imc_data_scvi():
    os.environ["SPATIALMUON_TEST"] = "analyses/scvi_analyses/imc_data_scvi.py"
    import analyses.scvi_analyses.imc_data_scvi


def test_visium_data_scvi():
    os.environ["SPATIALMUON_TEST"] = "analyses/scvi_analyses/visium_data_scvi.py"
    import analyses.scvi_analyses.visium_data_scvi


if __name__ == "__main__":
    if is_debug():
        pass
