#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu
#SBATCH --mail-type=END
#SBATCH --job-name=feat-alex

export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

srun python -u script/categorisation/features.py -start ${1} -stop ${2}