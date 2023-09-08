#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=END
#SBATCH --job-name=loc

export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

srun python -u script/localisation/evaluate_localisation.py -m -dto -dt 0.05 -ks 21 -ms 10
