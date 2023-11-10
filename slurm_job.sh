#!/bin/bash

#SBATCH -A GLEADALL-SL3-CPU
#SBATCH -p icelake

#SBATCH --job-name=mw922
#SBATCH --nodelist=cpu-q-456

#SBATCH --output=res_%j.out
#SBATCH --error=res_%j.err

#! How many nodes should be allocated? If not specified SLURM assumes 1 node.
#SBATCH --nodes=1

#! How many tasks will there be in total? By default SLURM will assume 1 task per node and 1 CPU per task.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76

#! How much memory in MB is required per node? Not setting this as here will lead to a default amount per task.
#! Setting a larger amount per task increases the number of CPUs.
##SBATCH --mem=15000

#! How much wallclock time will be required?
#SBATCH --time=12:00:00

#! Run your script with different arguments
#! srun python main.py --model_name "newnew" --LHD_configs 500 --emin 0 --emax 500 --total_cores_max 8 &
srun python main.py &

wait

