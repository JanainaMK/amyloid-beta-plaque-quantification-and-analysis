#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G
#SBATCH --mail-type=END
#SBATCH --job-name=cluster

export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

srun python -u script/categorisation/cluster.py -m mbkmeans -noc 5 -af