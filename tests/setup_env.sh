#!/bin/bash
set -e
source ~/.bashrc
conda remove -n ci_env --all -y
echo ">>>>>>>>>> creating empty env <<<<<<<<<<"
mamba create -n ci_env -y
echo ">>>>>>>>>> activating env <<<<<<<<<<"
conda activate ci_env
echo ">>>>>>>>>> installing most of the packages <<<<<<<<<<"
# mamba env update -n ci_env -f requirements_cuda.yml
mamba env update -n ci_env -f /data/l989o/deployed/a/requirements_cuda.yml
# echo ">>>>>>>>>> installing torch <<<<<<<<<<"
mamba install -c pytorch -c pyg -c conda-forge pytorch torchvision torchaudio cudatoolkit=10.2 pyg optuna optuna-dashboard pyro-ppl pytorch-lightning -y
# mamba install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
# mamba install -c pyg -c conda-forge pyg -y
# mamba install -c conda-forge optuna optuna-dashboard pyro-ppl pytorch-lightning -y
# mamba install pytorch==1.9.0 torchvision==0.8.1 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch -y
# echo ">>>>>>>>>> installing torch geometric <<<<<<<<<<"
python -c "import torch; print('torch version:', torch.__version__)"
python -c "import torch; print('cuda version:', torch.version.cuda)"
# python -c "import torch; assert torch.__version__ == '1.7.0'"
# python -c "import torch; assert torch.version.cuda == '10.2'"
echo ">>>>>>>>>> installing spatialmuon with develop <<<<<<<<<<"
pushd .
cd /data/l989o/deployed/spatialmuon
python setup.py develop
popd