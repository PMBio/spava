source ~/.bashrc
conda remove -n ci_env --all -y
mamba create -c conda-forge -c bioconda -n ci_env python=3.9 -y
conda activate ci_env
echo ">>>>>>>>>> installing all but torch <<<<<<<<<<"
mamba env update -n ci_env -f requirements_cuda.yml
echo ">>>>>>>>>> installing torch <<<<<<<<<<"
mamba install pytorch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch -y
echo ">>>>>>>>>> installing torch geometric <<<<<<<<<<"
python -c "import torch; print('torch version:', torch.__version__)"
python -c "import torch; print('cuda version:', torch.version.cuda)"
python -c "import torch; assert torch.__version__ == '1.7.0'"
python -c "import torch; assert torch.version.cuda == '10.2'"