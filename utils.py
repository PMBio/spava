import os
from joblib import Memory
from data2 import file_path
import numpy as np

f = file_path('joblib_cache_dir')
os.makedirs(f, exist_ok=True)
memory = Memory(cachedir=f)


def reproducible_random_choice(n: int, k: int):
    state = np.random.get_state()
    np.random.seed(42)
    ii = np.random.choice(n, k, replace=False)
    np.random.set_state(state)
    return ii
