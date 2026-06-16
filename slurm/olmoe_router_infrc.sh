#!/bin/bash

#SBATCH --job-name=OLMoE-1B-7B.router
#
#SBATCH -N1
#SBATCH -n9
#SBATCH --cpus-per-task=1
#SBATCH --account=pi_duc.tran
#SBATCH --time=00-20:00:00
#SBATCH --mem=120gb
#SBATCH --error=logs/OLMoE-1B-7B.router.err
#SBATCH --output=logs/OLMoE-1B-7B.router.log
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

echo "Starting smi-nvidia log ..."
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
           --format=csv -l 5 > ./logs/gpu_usage.csv &

echo "Starting inference task ..."
echo "Router is running at $HOSTNAME:$PORT"
rm -rf /home/duy.le004/phd/MoE/logs/packets/router/*
# python example.py
python -m eval.mmlu
echo "All jobs completed."