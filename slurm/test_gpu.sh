#!/bin/bash

#SBATCH --job-name=OLMoE-1B-7B.inference
#
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --account=pi_duc.tran
#SBATCH --time=00-20:00:00
#SBATCH --mem=80gb
#SBATCH --error=log/OLMoE-1B-7B.inference.err
#SBATCH --output=log/OLMoE-1B-7B.inference.log
#SBATCH --partition=DGXA100
#SBATCH --export=HOME
#
# specify gpu type and quantity.  Uncomment the option you
# want
##SBATCH --gres=gpu:h200:1
##SBATCH --gres=gpu:3g.71gb:1
##SBATCH --gres=gpu:1g.18gb:1
#SBATCH --gres=gpu:A100:1

# Optional - enforce GPU/CPU affinity
##SBATCH --gres-flags=enforce-binding

# Optional email alerts
##SBATCH --mail-type=ALL
##SBATCH --mail-user=duy.le004@umb.edu

# source the local environment if --export=HOME 

echo "Activating Conda ..."
source /share/apps/linux-ubuntu20.04-zen2/anaconda3-2021.05/etc/profile.d/conda.sh
conda activate perft-moe
python --version
PYTHON_BIN=$(which python)
echo "Using PYTHON_BIN=$PYTHON_BIN"

srun --ntasks=64 --cpu-bind=cores --export=ALL "$PYTHON_BIN" ./slurm/worker_probe.py

# Diagnostic/Logging Information
echo "Finish Run"
echo "end time is `date`"