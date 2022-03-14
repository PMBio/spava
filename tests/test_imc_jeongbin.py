import os
from tests.testing_utils import is_debug


def test_preprocess_imc_jeongbin_data():
    os.environ["SPATIALMUON_TEST"] = "datasets/imc_jeongbin_data.py"
    import datasets.imc_jeongbin_data


if __name__ == "__main__":
    if is_debug():
        pass
