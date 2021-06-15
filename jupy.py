from splits import *
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('ds/spatial_uzh_processed/a/', exist_ok=True)


def plt_show(plt, filename):
    f = os.path.join('ds/spatial_uzh_processed/a', filename)
    plt.savefig(f)
    print(f'saving figure in {f}')
    plt.show()
