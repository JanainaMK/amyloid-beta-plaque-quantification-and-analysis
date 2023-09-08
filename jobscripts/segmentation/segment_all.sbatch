#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --job-name=Abeta-segment

export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

srun python -u script/segmentation/segment_all.py