import os
from joblib import Memory
from data2 import file_path
import numpy as np
import matplotlib
import sys

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
    # print(f"CI_TEST = {CI_TEST}")

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
    return COMPUTE, PLOT, DEBUG
