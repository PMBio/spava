# change username, dataset location and virtualenv according to experiment

source ~/.bashrc

#export PATH=/bin:$PATH
#export OMP_NUM_THREADS=1 # needed to be able to run batchgenerators with new numpy versions. Use this!
#
##module load python/3.7.0 # load python module. If you don't do this then you have no python
#
##source /dkfz/cluster/virtualenvs/{USERNAME}/{YOUR_ENV}/bin/activate # activate my python virtual environment
## or, if you do not use a shared virtualenv:
## source /home/{PATH_TO_ENV}/bin/activate
## this part only makes sense if you have one virtual environment for all your experiments. If you have multiple, please see the runner script section below
#
#CUDA=10.2 # or 11.0
#export PATH=/usr/local/lib:$PATH
#export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
#export CPATH=/usr/local/lib:$CPATH
#export PATH=/usr/local/cuda-${CUDA}/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA}/lib64:$LD_LIBRARY_PATH
#export CUDA_HOME=/usr/local/cuda-${CUDA}
#export CUDA_CACHE_DISABLE=1

# export DATASET_LOCATION=/dkfz/cluster/gpu/data/OE0540/l989o/raw/spatial_uzh/spatial_uzh_processed/a
# export EXPERIMENT_LOCATION=/dkfz/cluster/gpu/checkpoints/OE0540/l989o/{EXP_RESULTS}

conda activate ci_env
python -c 'import torch; torch.cuda.is_available()'