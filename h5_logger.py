import os
from typing import Dict, Union

import h5py
import numpy as np
import torch


class H5Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file

    def clear(self):
        # if os.path.isfile(self.log_file):
        #     print(f'deleting {self.log_file}')
        #     os.remove(self.log_file)
        with h5py.File(self.log_file, 'w'):
            pass

    def log(self, epoch: int, data: Dict[str, Union[int, float, np.array, np.ndarray, torch.Tensor]]):
        f5 = h5py.File(self.log_file, 'a')

        for key, value in data.items():
            internal_path = f'epoch{epoch}/{key}'
            if type(value) == int:
                value = np.array(value, dtype=np.int32)
            elif type(value) == float:
                value = np.array(value, dtype=np.float)
            elif type(value) == torch.Tensor:
                value = value.cpu().detach().numpy()
            f5[internal_path] = value
        f5.close()
