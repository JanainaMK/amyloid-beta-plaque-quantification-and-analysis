#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --job-name=loc_series


export PYTHONPATH="${PYTHONPATH}:/tudelft.net/staff-umbrella/NightId/chieldevries"

OUTFOLDER="result/outfiles_loc"

START=${1}
STOP=${2}
CASES=${3}

srun --output="${OUTFOLDER}/${SLURM_JOBID}-loc-${START}-${STOP}-${CASES}.out" python -u script/localisation/loc_series.py --start ${START} --stop ${STOP} --cases ${CASES}
