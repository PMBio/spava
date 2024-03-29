#!/bin/bash
set -e
echo `pwd`
source ~/.bashrc
conda remove -n scvi_env --all -y
echo ">>>>>>>>>> creating empty env <<<<<<<<<<"
mamba create -n scvi_env python=3.8 -y
echo ">>>>>>>>>> activating env <<<<<<<<<<"
conda activate scvi_env
echo ">>>>>>>>>> installing most of the packages <<<<<<<<<<"
pip install scvi-tools scanpy anndata pytest rtree shapely trimesh geopandas colorama scikit-image torchvision matplotlib-scalebar opencv-python jupyter jupyterlab jupytext louvain datashader dill pytest-randomly
pushd .
cd ../../spatialmuon
python setup.py develop
popd