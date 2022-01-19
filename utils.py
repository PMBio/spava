import os
from joblib import Memory
from utils import file_path
import numpy as np
import matplotlib
import sys
import pathlib

f = file_path("joblib_cache_dir")
os.makedirs(f, exist_ok=True)
memory = Memory(location=f)


def reproducible_random_choice(n: int, k: int):
    state = np.random.get_state()
    np.random.seed(42)
    ii = np.random.choice(n, k, replace=False)
    np.random.set_state(state)
    return ii


def setup_ci(name):
    CI_TEST = "CI_TEST" in os.environ
    NOTEBOOK_EXPORTER = "NOTEBOOK_EXPORTER" in os.environ
    assert not (CI_TEST and NOTEBOOK_EXPORTER)

    if CI_TEST:
        if sys.gettrace() is None:
            matplotlib.use("Agg")
        COMPUTE = True
        PLOT = True
        TEST = True
        NOTEBOOK = False
    elif NOTEBOOK_EXPORTER:
        COMPUTE = True
        PLOT = True
        TEST = False
        NOTEBOOK = True
    elif name == '__main__':
        COMPUTE = True
        PLOT = True
        TEST = False
        NOTEBOOK = False
    else:
        COMPUTE = False
        PLOT = False
        TEST = False
        NOTEBOOK = False
    print(f'COMPUTE = {COMPUTE}, PLOT = {PLOT}, TEST = {TEST}, NOTEBOOK = {NOTEBOOK}')
    return COMPUTE, PLOT, TEST, NOTEBOOK

try:
    current_file_path = pathlib.Path(__file__).parent.absolute()

    def file_path(f):
        if os.path.exists('data'):
            return os.path.join(current_file_path, "data/spatial_uzh_processed/a", f)
        else:
            raise NameError

except NameError:
    print("setting data path manually")

    def file_path(f):
        return os.path.join("/data/l989o/data/basel_zurich/spatial_uzh_processed/a", f)
