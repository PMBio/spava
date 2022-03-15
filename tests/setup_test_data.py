##
import os
import shutil
import spatialmuon as smu
import numpy as np
from utils import file_path

# the folder is called 'a'
a_folder = file_path(".")
spatial_uzh_processed_folder = os.path.dirname(os.path.dirname(a_folder))
assert os.path.join(spatial_uzh_processed_folder, "a/check") == file_path("check")
assert os.path.isdir(spatial_uzh_processed_folder)
a_test_folder = os.path.join(spatial_uzh_processed_folder, "a_test")

##
if os.path.isdir(a_test_folder):
    shutil.rmtree(a_test_folder)

##
smu_des_imc_folder = os.path.join(a_test_folder, "spatialmuon/imc")
smu_des_visium_mousebrain_folder = os.path.join(
    a_test_folder, "spatialmuon/visium_mousebrain"
)
os.makedirs(smu_des_imc_folder, exist_ok=True)
os.makedirs(smu_des_visium_mousebrain_folder, exist_ok=True)
##

imc_files = [
    "BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_107_114_X13Y4_219_a0_full.tiff",
    "BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_106_18_X13Y5_248_a0_full.tiff",
    "BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_123_86_X15Y3_203_a0_full.tiff",
]

for imc in imc_files:
    imc_smu = imc.replace(".tiff", ".h5smu")
    src_f = os.path.join(spatial_uzh_processed_folder, f"a/spatialmuon/imc/{imc_smu}")
    assert os.path.isfile(src_f)
    des_f = os.path.join(smu_des_imc_folder, imc_smu)
    shutil.copy(src_f, des_f)
    s = smu.SpatialMuData(backing=des_f, backingmode="r+")
    n = 100
    new_masks_x = s["imc"]["masks"].masks.X[:n, :n]
    new_regions = smu.Regions(
        masks=smu.RasterMasks(X=new_masks_x), anchor=s["imc"]["masks"].anchor
    )
    channels = np.array([8, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert len(channels) == 10
    new_ome = s["imc"]["ome"].X[:n, :n, channels]
    new_raster = smu.Raster(X=new_ome, anchor=s["imc"]["ome"].anchor)
    sm = smu.SpatialModality()
    sm["masks"] = new_regions
    sm["ome"] = new_raster
    del s["imc"]
    s["imc"] = sm
    s.backing.close()

##
visium_mousebrain_src = "/data/l989o/deployed/spatialmuon/tests/data/small_visium.h5smu"
visium_mousebrain_des = os.path.join(smu_des_visium_mousebrain_folder, "visium.h5smu")
shutil.copy(visium_mousebrain_src, visium_mousebrain_des)
# s = smu.SpatialMuData(visium_mousebrain_des)
# s
##
print("done")
