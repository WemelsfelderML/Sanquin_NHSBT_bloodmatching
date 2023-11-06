#!/bin/bash

#SBATCH --job-name=BloodMatch_mw922
#SBATCH -A GLEADALL-SL3-CPU
#SBATCH -p cclake

#SBATCH --output=res_%j.out
#SBATCH --error=res_%j.err

#! How many nodes should be allocated? If not specified SLURM assumes 1 node.
#SBATCH --nodes=1

#! How many tasks will there be in total? By default SLURM will assume 1 task per node and 1 CPU per task.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#! How much memory in MB is required per node? Not setting this as here will lead to a default amount per task.
#! Setting a larger amount per task increases the number of CPUs.
##SBATCH --mem=3000

#! How much wallclock time will be required?
#SBATCH --time=1:00:00

#! Run your script with different arguments
#! srun python main.py --model_name "ranged" --LHD_configs 100 --emin 0 --emax 100 --total_cores_max 8 &
# srun python main.py --model_name "newnew" --LHD_configs 500 --emin 0 --emax 500 --total_cores_max 8 &
srun python main.py &

# for i in {0..499}; do
#   emin=$i
#   emax=$((i+1))
#   srun python main.py --model_name "newnew" --LHD_configs 500 --emin "$emin" --emax "$emax" --total_cores_max 1 &
# done

wait

