##
import os

import colorama
import h5py
import matplotlib.pyplot as plt
import numpy as np
import spatialmuon as smu
from tqdm.auto import tqdm

from datasets.visium_endometrium import (
    get_smu_file,
    get_split_bimap,
    visium_endrometrium_samples,
)
from utils import get_execute_function, file_path, parse_flags

e_ = get_execute_function()

flags = parse_flags(default={"TILE_SIZE": 32})

plt.style.use("dark_background")

##
if e_():
    print(f"{colorama.Fore.MAGENTA}extracting tiles{colorama.Fore.RESET}")
    d = file_path("visium_endometrium")
    os.makedirs(d, exist_ok=True)

    ss = {}
    tiless = {}
    for sample in tqdm(visium_endrometrium_samples, desc="extracting tiles"):
        s = get_smu_file(sample, read_only=True)
        ss[sample] = s
        if flags["TILE_SIZE"] == "large":
            print(
                f"{colorama.Fore.MAGENTA}55um = {s['visium']['medium_res'].anchor.inverse_transform_length(55)} "
                f"pixels{colorama.Fore.RESET}"
            )
            raster = s["visium"]["medium_res"]
            tile_size = 128
        else:
            raster = s["visium"]["image"]
            tile_size = flags["TILE_SIZE"]
        tiles = smu.Tiles(
            raster=raster,
            masks=s["visium"]["processed"].masks,
            tile_dim_in_pixels=tile_size,
        )
        if sample == visium_endrometrium_samples[0]:
            tiles._example_plot()
        tiless[sample] = tiles
        s.backing.close()

    f = file_path(f"visium_endometrium/tiles_{flags['TILE_SIZE']}.hdf5")
    with h5py.File(f, "w") as f5:
        for split in tqdm(
            ["train", "validation", "test"], desc="split", position=0, leave=True
        ):
            map_left, map_right = get_split_bimap(split)
            split_tiles = []
            for sample, index_in_sample in map_left.values():
                split_tiles.append(tiless[sample].tiles[index_in_sample])
            f5[f"{split}"] = split_tiles

    for sample in visium_endrometrium_samples:
        s = ss[sample]
        s.backing.close()
    print("done")
