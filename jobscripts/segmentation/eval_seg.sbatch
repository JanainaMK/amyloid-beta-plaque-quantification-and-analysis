#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --job-name=Abeta-eval

export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

srun python -u script/segmentation/evaluate.py models/2022-03-13-unet-16x-bs16-ps256-lr0.001/2022-03-13-unet-16x-bs16-ps256-lr0.001-e0v49.pt