import os
import numpy as np
import matplotlib
import sys
import pathlib
import colorama
import inspect
from typing import Dict


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
                or "SPATIALMUON_NOTEBOOK" in os.environ
                and caller_filename.startswith("/tmp/ipykernel_")
            )

    return execute_


path_debug = None


def print_and_return(f):
    color = colorama.Fore.GREEN if "/a_test/" in f else colorama.Fore.CYAN
    # a "color change" means that there is a bug and somewhere get_execute_function() is called both before and
    # after 'SPATIALMUON_TEST' or 'SPATIALMUON_NOTEBOOK' are set, leading to mixing test and default paths
    global path_debug
    if path_debug is None:
        path_debug = color
    else:
        assert path_debug == color
    PRINT = False
    # PRINT = True
    if PRINT:
        print(f"{color}{f}{colorama.Fore.RESET}")
    return f


try:
    current_file_path = pathlib.Path(__file__).parent.absolute()
    if not os.path.exists("data"):
        raise NameError()

    def file_path(f):
        if "SPATIALMUON_TEST" not in os.environ:
            return print_and_return(
                os.path.join(current_file_path, "data/spatial_uzh_processed/a", f)
            )
        else:
            return print_and_return(
                os.path.join(current_file_path, "data/spatial_uzh_processed/a_test", f)
            )

except NameError:
    print("setting data path manually")

    def file_path(f):
        if "SPATIALMUON_TEST" not in os.environ:
            return print_and_return(
                os.path.join("/data/l989o/data/basel_zurich/spatial_uzh_processed/a", f)
            )
        else:
            return print_and_return(
                os.path.join(
                    "/data/l989o/data/basel_zurich/spatial_uzh_processed/a_test", f
                )
            )


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


def validate_and_cast_flags(flags: Dict, cast: bool):
    validate_type_functions = {
        "MODEL_NAME": lambda s: True,
        "TILE_SIZE": lambda s: s.isdigit(),
        "GRAPH_SIZE": lambda s: s.isdigit(),
        "PCA": lambda s: s in ["True", "False"],
    }
    cast_functions = {
        "MODEL_NAME": lambda s: s,
        "TILE_SIZE": int,
        "GRAPH_SIZE": int,
        "PCA": lambda s: s == 'True',
    }
    validation_functions = {
        "MODEL_NAME": lambda s: s
        in ["expression_vae", "image_expression_vae", "image_expression_pca_vae", "image_expression_conv_vae"],
        "TILE_SIZE": lambda s: 32 <= int(s) <= 256,
        "GRAPH_SIZE": lambda s: 3 <= int(s) <= 50,
        "PCA": lambda s: True,
    }
    assert set(validate_type_functions.keys()) == set(cast_functions.keys())
    assert set(cast_functions.keys()) == set(validation_functions.keys())
    for flag in flags.keys():
        assert flag in validation_functions

    if cast:
        for value in flags.values():
            assert type(value) == str
        for flag, value in flags.items():
            validate_type_functions[flag](value)
            flags[flag] = cast_functions[flag](value)

    for flag, value in flags.items():
        assert validation_functions[flag](value)


def parse_flags(default: Dict):
    if "SPATIALMUON_FLAGS" in os.environ:
        flags = os.environ["SPATIALMUON_FLAGS"]
    else:
        validate_and_cast_flags(default, cast=False)
        return default

    d = {}
    flags = flags.split(",")
    for flag in flags:
        assert flag.count("=") == 1
        k, v = flag.split("=")
        d[k] = v

    # assert set(default.keys()).issuperset(set(d.keys()))
    # for k_def, v_def in default.items():
    #     if k_def not in d:
    #         d[k_def] = v_def

    validate_and_cast_flags(d, cast=True)
    return d
