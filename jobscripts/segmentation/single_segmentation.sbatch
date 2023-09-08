#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=4-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=1536M
#SBATCH --mail-type=END
#SBATCH --gpus-per-task=1
#SBATCH --job-name=Abeta

export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

srun --exclusive --ntasks=1 --gpus-per-task="$SLURM_GPUS_PER_TASK" --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem-per-cpu="$SLURM_MEM_PER_CPU" --output="${SLURM_JOBID}-lr-0.00005.out" python -u script/segmentation/train_interval.py 2023-03-15 -e 4 -lr 0.00005 -l models/2022-03-13-unet-16x-bs16-ps256-lr5e-05/2022-03-13-unet-16x-bs16-ps256-lr5e-05-e0v49.pt

