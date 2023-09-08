#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END
#SBATCH --job-name=cluster-report

export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

NAME=${1} # "kmeans-dmcosine-nc5-alex"
mkdir "result/cluster-report/$NAME"

srun --output "result/cluster-report/$NAME/report.out" python -u script/categorisation/cluster_report.py ${NAME}.npy

