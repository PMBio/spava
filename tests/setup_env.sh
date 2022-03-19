#!/bin/bash
set -e
echo `pwd`
source ~/.bashrc
conda remove -n ci_env --all -y
echo ">>>>>>>>>> creating empty env <<<<<<<<<<"
mamba create -n ci_env -y
echo ">>>>>>>>>> activating env <<<<<<<<<<"
conda activate ci_env
echo ">>>>>>>>>> installing most of the packages <<<<<<<<<<"
mamba env update -n ci_env -f ../requirements_cuda.yml
# echo ">>>>>>>>>> installing torch-related packages <<<<<<<<<<"
#mamba install -c pytorch -c pyg -c conda-forge pytorch torchvision torchaudio cudatoolkit=10.2 pyg optuna optuna-dashboard pyro-ppl pytorch-lightning -y
mamba install -c pytorch -c conda-forge pytorch==1.9.0 torchvision torchaudio cudatoolkit=10.2 optuna optuna-dashboard pyro-ppl pytorch-lightning -y
mamba install -c pyg -c pytorch pyg -y
echo ">>>>>>>>>> installing spatialmuon with develop <<<<<<<<<<"
pushd .
cd /data/l989o/deploayed/spatialmuon
python setup.py develop
popd
pip install matplotlib-scalebar louvain pytest-randomly
