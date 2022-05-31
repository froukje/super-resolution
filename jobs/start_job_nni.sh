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
configdir=/work/ka1176/frauke/super-resolution/nni/config


#PYTHONPATH=$PYTHONPATH:"$codedir"
#export PYTHONPATH

port=$((8080 + $RANDOM % 10000))
echo $port
nnictl create -c /work/ka1176/frauke/super-resolution/nni/config/config.yml -p $port|| nnictl create -c /work/ka1176/frauke/super-resolution/nni/config/config.yml -p $port|| nnictl create -c /work/ka1176/frauke/super-resolution/nni/config/config.yml -p $port|| nnictl create -c /work/ka1176/frauke/super-resolution/nni/config/config.yml -p $port

#nnictl create -c $configdir/config.yml -p $port || nnictl create -c $configdir/config.yml -p $port || nnictl create -c $configdir/config.yml -p $port || nnictl create -c $configdir/config.yml -p $port
sleep 23h 50m
nnictl stop

