#!/bin/bash
#SBATCH --job-name=sr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=ka1176
#SBATCH --output=sr_out.o%j
#SBATCH --error=sr_err.e%j
#SBATCH --reservation=GPUtest

conda init
source ~/.bashrc
conda activate super-res
echo "conda env activated"

# Run script
codedir=/work/ka1176/frauke/super-resolution
datadir=/work/ka1176/frauke/super-resolution/data

#PYTHONPATH=$PYTHONPATH:"$codedir"
#export PYTHONPATH

python3 $codedir/srresnet.py --batch-size 8 --n-channels 8 --gpus 1 --data-dir $datadir --output-path $datadir/pred.h5 
