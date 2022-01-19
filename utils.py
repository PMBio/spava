import os
from joblib import Memory
from data2 import file_path
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

    if name == "__main__":
        PLOT = True
        COMPUTE = True
        DEBUG = True
    elif CI_TEST:
        if sys.gettrace() is None:
            matplotlib.use("Agg")
        PLOT = True
        COMPUTE = True
        DEBUG = False
    else:
        PLOT = False
        COMPUTE = False
        DEBUG = False

    if PLOT:
        COMPUTE = True
    print(f'COMPUTE = {COMPUTE}, PLOT = {PLOT}, DEBUG = {DEBUG}')
    return COMPUTE, PLOT, DEBUG

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
