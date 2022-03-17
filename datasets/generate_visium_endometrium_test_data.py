##
import os
import shutil
import time

import matplotlib.pyplot as plt
import spatialmuon as smu
import numpy as np
import pandas as pd
from utils import file_path
from tqdm.auto import tqdm

# the folder is called 'a'
a_folder = file_path(".")
spatial_uzh_processed_folder = os.path.dirname(os.path.dirname(a_folder))

##
visium_endrometrium_samples = ["152806", "152807", "152810", "152811"]
for sample in tqdm(visium_endrometrium_samples):
    visium_endometrium_src = os.path.join(
        spatial_uzh_processed_folder,
        f"a/spatialmuon/visium_endometrium/{sample}.h5smu",
    )
    visium_endometrium_des = os.path.join(
        f'/data/spatialmuon/datasets/visium_endometrium/smu/subsampled/{sample}.h5smu'
    )
    os.makedirs(os.path.dirname(visium_endometrium_des), exist_ok=True)
    shutil.copy(visium_endometrium_src, visium_endometrium_des)

    s = smu.SpatialMuData(backing=visium_endometrium_des)
    regions0 = s["visium"]["expression"]
    regions1 = s["visium"]["mean_nUMI_factors"]
    raster0 = s["visium"]["image"]
    raster1 = s["visium"]["medium_res"]
    raster2 = s["visium"]["hires_image"]

    x, y = np.median(regions0.transformed_centers, axis=0)
    r = 70
    bb = smu.BoundingBox(x0=x - r, x1=x + r, y0=y - r, y1=y + r)
    cropped_masks = regions0.masks.crop(bb)
    labels = cropped_masks.masks_labels
    original_labels = regions0.masks.masks_labels

    ##
    NEW_N_VAR = 50
    def new_regions_from_labels(regions, labels):
        xx = regions.X[...]
        if type(xx) != np.ndarray:
            new_x = xx.todense().copy()
        else:
            new_x = xx.copy()
        new_x = new_x[: len(labels), :NEW_N_VAR]
        r = smu.Regions(
            X=new_x,
            masks=cropped_masks.clone(),
            anchor=regions.anchor.clone(),
            var=regions.var[:NEW_N_VAR],
        )
        return r

    new_regions0 = new_regions_from_labels(regions0, labels)
    new_regions1 = new_regions_from_labels(regions1, labels)

    ##
    def get_new_raster(raster, bb):
        new_raster = raster.clone()
        new_raster.crop(bb)
        return new_raster

    new_raster0 = get_new_raster(raster0, bb)
    new_raster1 = get_new_raster(raster1, bb)
    new_raster2 = get_new_raster(raster2, bb)

    if False:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        new_raster0.plot(ax=ax)
        new_regions1.masks.plot(fill_colors="k", ax=ax)
        plt.show()
    ##
    del s["visium"]["expression"]
    s["visium"]["expression"] = new_regions0

    del s["visium"]["mean_nUMI_factors"]
    s["visium"]["mean_nUMI_factors"] = new_regions1

    del s["visium"]["image"]
    s["visium"]["image"] = new_raster0

    del s["visium"]["medium_res"]
    s["visium"]["medium_res"] = new_raster1

    del s["visium"]["hires_image"]
    s["visium"]["hires_image"] = new_raster2

    ##
    print('repacking...')
    start = time.time()
    s.repack()
    print(f'repacking: {time.time() - start}')
print('done')
