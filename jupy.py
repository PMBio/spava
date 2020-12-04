from splits import *
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('data/spatial_uzh_processed/a/', exist_ok=True)


def plt_show(plt, filename):
    f = os.path.join('data/spatial_uzh/processed/a', filename)
    plt.savefig(filename)
    plt.show()
