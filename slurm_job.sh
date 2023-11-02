#!/bin/bash

#SBATCH --job-name=BloodMatch_Merel
#SBATCH -A mw922
#SBATCH -p cclake

#! How many nodes should be allocated? If not specified SLURM assumes 1 node.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

#! How many tasks will there be in total? By default SLURM will assume 1 task per node and 1 CPU per task.
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1

#! How much memory in MB is required per node? Not setting this as here will lead to a default amount per task.
#! Setting a larger amount per task increases the number of CPUs.
##SBATCH --mem=

#! How much wallclock time will be required?
#SBATCH --time=02:00:00

#! Run your script with different arguments
srun python main.py --model_name "ranged" --LHD_configs 100 --emin 0 --emax 100 --total_cores_max 8 &
srun python main.py --model_name "newnew" --LHD_configs 500 --emin 0 --emax 500 --total_cores_max 8 &
#! srun git add -f results &
#! srun git add -f optimize_params &

wait
