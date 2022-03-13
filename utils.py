import os
from joblib import Memory
import numpy as np
import matplotlib
import sys
import pathlib
import inspect


def reproducible_random_choice(n: int, k: int):
    state = np.random.get_state()
    np.random.seed(42)
    ii = np.random.choice(n, k, replace=False)
    np.random.set_state(state)
    return ii


def get_execute_function():
    def adjust_path(f):
        if os.path.isfile(f):
            parent_path = str(pathlib.Path(__file__).parent.absolute()) + "/"
            assert f.startswith(
                parent_path
            ), f'f = "{f}", parent_path = "{parent_path}"'
            f = f.replace(parent_path, "")
            return f
        else:
            return f

    def execute_():
        if (
            "SPATIALMUON_NOTEBOOK" not in os.environ
            and "SPATIALMUON_TEST" not in os.environ
        ):
            return False
        else:
            if "SPATIALMUON_NOTEBOOK" in os.environ:
                assert "SPATIALMUON_TEST" not in os.environ
                target = os.environ["SPATIALMUON_NOTEBOOK"]
            else:
                if sys.gettrace() is None:
                    matplotlib.use("Agg")
                target = os.environ["SPATIALMUON_TEST"]
            caller_filename = inspect.stack()[1].filename
            caller_filename = adjust_path(caller_filename)
            print(f"target = {target}, caller_filename = {caller_filename}")
            return (
                target == caller_filename
                or "SPATIALMUON_NOTEBOOK" in os.environ
                and caller_filename.startswith("<ipython-input-")
            )

    return execute_


try:
    current_file_path = pathlib.Path(__file__).parent.absolute()
    if not os.path.exists("data"):
        raise NameError()

    def file_path(f):
        if "SPATIALMUON_TEST" not in os.environ:
            return os.path.join(current_file_path, "data/spatial_uzh_processed/a", f)
        else:
            return os.path.join(
                current_file_path, "data/spatial_uzh_processed/a_test", f
            )

except NameError:
    print("setting data path manually")

    def file_path(f):
        if "SPATIALMUON_TEST" not in os.environ:
            return os.path.join(
                "/data/l989o/data/basel_zurich/spatial_uzh_processed/a", f
            )
        else:
            return os.path.join(
                "/data/l989o/data/basel_zurich/spatial_uzh_processed/a_test", f
            )


f = file_path("joblib_cache_dir")
os.makedirs(f, exist_ok=True)
memory = Memory(location=f)


def get_bimap(names_length_map):
    map_left = {}
    map_right = {}
    i = 0
    for name, length in names_length_map.items():
        for j in range(length):
            map_left[i] = (name, j)
            map_right[(name, j)] = i
            i += 1
    return map_left, map_right


def print_corrupted_entries_hash(corrupted_entries: np.ndarray, split: str):
    h = np.sum(np.concatenate(np.where(corrupted_entries == 1)))
    print(f"corrupted entries hash ({split}):", h)
