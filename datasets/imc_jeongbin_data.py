##
import spatialmuon as smu
from utils import get_execute_function, file_path

e_ = get_execute_function()

##
if e_():
    f = '/data/spatialmuon/datasets/imc_jeongbin/smu/imc.h5smu'
    s = smu.SpatialMuData(f)
    print(s)