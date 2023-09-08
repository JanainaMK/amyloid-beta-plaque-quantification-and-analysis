#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --job-name=Abeta-segment

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

srun python -u src/segmentation/segment.py

