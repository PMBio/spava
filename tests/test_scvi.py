import os
from tests.testing_utils import is_debug


def test_imc_scvi():
    os.environ["SPATIALMUON_TEST"] = "analyses/scvi_analyses/imc_scvi.py"
    import analyses.scvi_analyses.imc_scvi


def test_visium_mousebrain_scvi():
    os.environ["SPATIALMUON_TEST"] = "analyses/scvi_analyses/visium_mousebrain_scvi.py"
    import analyses.scvi_analyses.visium_mousebrain_scvi


if __name__ == "__main__":
    if is_debug():
        pass
