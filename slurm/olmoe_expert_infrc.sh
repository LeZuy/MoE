#!/bin/bash

# Sample slurm submission script for the Chimera
# compute cluster
# Lines beginning with # are comments, and will be ignored by
# the interpreter.  Lines beginning with #SBATCH are
# directives to the scheduler.  These in turn can be
# commented out by adding a second # (e.g. ##SBATCH lines
# will not be processed by the scheduler).
#
#
# set name of job
#SBATCH --job-name=OLMoE-1B-7B.expert
#
# set the number of processors/tasks needed
##SBATCH -n 4
# for hyperthreaded,shared memory jobs, set 1 task, 1 node,
# and set --cpus-per-task to total number of threads
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1

# set the number of Nodes needed.  Set to 1 for shared 
# memory jobs
#SBATCH -N 1

#set an account to use
#if not used then default will be used
# for scavenger users, use this format:
##SBATCH --account=pi_first.last
# for contributing users, use this format:
##SBATCH --account=<deptname|lastname>

# set max wallclock time  DD-HH:MM:SS
#SBATCH --time=00-10:00:00

# set a memory request
#SBATCH --mem=160gb

# Set filenames for stdout and stderr.  %j can be used
# for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=logs/OLMoE-1B-7B.expert.err
#SBATCH --output=logs/OLMoE-1B-7B.expert.log
#

# set the partition where the job will run.  Multiple partitions can
# be specified as a comma separated list
# Use command "sinfo" to get the list of partitions
##SBATCH --partition=Intel6240
#SBATCH --partition=Intel6240,Intel6248,DGXA100
# restricting inheritance of environment variables is
# required for chimera12 and 13:
# if this option is used, source /etc/profile below.
#SBATCH --export=HOME

#Optional
# mail alert at start, end and/or failure of execution
# see the sbatch man page for other options
##SBATCH --mail-type=ALL
# send mail to this address
##SBATCH --mail-user=first.last@umb.edu

# Put your job commands here, including loading any needed
# modules or diagnostic echos.

# this job simply reports the hostname and sleeps for
# two minutes

# source the local profile.  This is recommended in
# conjunction with the --export=HOME or --export=NONE
# sbatch options.
. /etc/profile

echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

echo "Activating Conda ..."
source /share/apps/linux-ubuntu20.04-zen2/anaconda3-2021.05/etc/profile.d/conda.sh
conda activate perft-moe

echo "Starting inference task ..."

for i in {0..7}; do
  rm -rf /home/duy.le004/phd/MoE/logs/packets/agent_$i/*
done 

echo "Starting expert agents ..."
srun -N1 -n8 -c1 --cpu-bind=cores \
  /bin/bash -lc '
    ID=$SLURM_PROCID
    HOST=$(hostname)

    exec /home/duy.le004/.conda/envs/perft-moe/bin/python -m network.expert_agent \
      --id $ID \
      --host $HOST
  '
  
echo "All jobs completed."

# Diagnostic/Logging Information
echo "Finish Run"
echo "end time is `date`"