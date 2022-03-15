##
import os

import colorama
import h5py
import matplotlib.pyplot as plt
import numpy as np
import spatialmuon as smu
from tqdm.auto import tqdm

from datasets.imc import get_smu_file, get_split
from utils import get_execute_function, file_path

# os.environ['SPATIALMUON_TEST'] = 'aaa'
# os.environ['SPATIALMUON_NOTEBOOK'] = 'aaa'
e_ = get_execute_function()
# matplotlib.use('module://backend_interagg')

plt.style.use("dark_background")

##
if e_():
    print(f"{colorama.Fore.MAGENTA}extracting tiles{colorama.Fore.RESET}")
    scaling_factors = get_smu_file(split="train", index=0, read_only=True)["imc"][
        "transformed_mean"
    ].uns["scaling_factors"][...]
    d = file_path("imc/")
    os.makedirs(d, exist_ok=True)
    f = file_path("imc/imc_tiles.hdf5")
    with h5py.File(f, "w") as f5:
        for split in tqdm(
            ["train", "validation", "test"], desc="split", position=0, leave=True
        ):
            for index in tqdm(
                range(len(get_split(split))), desc="slide", position=0, leave=True
            ):
                s = get_smu_file(split=split, index=index, read_only=True)
                filename = os.path.basename(s.backing.filename)
                raster = s["imc"]["ome"]
                x = raster.X[...]
                ##
                tiles = smu.Tiles(
                    raster=raster,
                    masks=s["imc"]["masks"].masks,
                    tile_dim_in_pixels=32,
                )
                # if p_:
                #     tiles._example_plot()
                f5[f"{split}/{filename}/raster"] = tiles.tiles
                ##
                masks_tiles = smu.Tiles(
                    masks=s["imc"]["masks"].masks,
                    tile_dim_in_pixels=32,
                )
                # if p_:
                #     masks_tiles._example_plot()
                ##
                f5[f"{split}/{filename}/masks"] = masks_tiles.tiles
                s.backing.close()
