##
import os

import colorama
import h5py
import matplotlib.pyplot as plt
import numpy as np
import spatialmuon as smu
from tqdm.auto import tqdm

from datasets.visium_mousebrain import get_smu_file, get_split_indices
from utils import get_execute_function, file_path, parse_flags

e_ = get_execute_function()

flags = parse_flags(default={'TILE_SIZE': 64})
tile_size = flags['TILE_SIZE']

plt.style.use("dark_background")

##
if e_():
    print(f"{colorama.Fore.MAGENTA}extracting tiles{colorama.Fore.RESET}")
    d = file_path("visium_mousebrain")
    os.makedirs(d, exist_ok=True)
    f = file_path(f"visium_mousebrain/tiles_{tile_size}.hdf5")
    s = get_smu_file(read_only=True)
    tiles = smu.Tiles(
        raster=s["visium"]["image"],
        masks=s["visium"]["processed"].masks,
        tile_dim_in_pixels=tile_size,
    )
    # tiles._example_plot()
    with h5py.File(f, "w") as f5:
        for split in tqdm(
            ["train", "validation", "test"], desc="split", position=0, leave=True
        ):
            indices = get_split_indices(split)
            split_tiles = [tiles.tiles[i] for i in indices]
            f5[f"{split}"] = split_tiles
    s.backing.close()
