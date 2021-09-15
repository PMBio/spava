##
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import torch
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import matplotlib.cm

from data2 import PerturbedRGBCells, PerturbedCellDataset, IndexInfo
import matplotlib.pyplot as plt
from models.ah_expression_vaes_lightning import VAE as ExpressionVAE, get_loaders
import scanpy as sc
import anndata as ad
import seaborn as sns
import pandas as pd
import optuna
from analyses.essentials import *
from utils import memory

m = __name__ == "__main__"

##
if m:
    SPLIT = "train"
    ds = PerturbedRGBCells(split=SPLIT)

    PERTURB = True
    if PERTURB:
        ds.perturb()

##
if m:
    pass
# the models: 49 original, 62 perturbed
