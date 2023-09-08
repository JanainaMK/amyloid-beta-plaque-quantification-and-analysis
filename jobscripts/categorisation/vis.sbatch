#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-type=END
#SBATCH --job-name=cluster-report

export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

srun python -u script/categorisation/visualisation.py result/cluster-report/sample_plaques-alex-dmcosine-e0.100-ms5 -noc 8 -ipc 10