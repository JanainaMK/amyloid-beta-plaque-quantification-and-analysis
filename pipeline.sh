#!/bin/bash
export PROJECT_PATH="/path/to/project"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"

# start and stop file indices
START_IDX=0
STOP_IDX=10

SOURCE_FOLDER1='dataset/Abeta images 100+'
SOURCE_FOLDER2='dataset/Abeta images AD cohort'
TARGET_FOLDER='result/images'

################
# Preparation #
################

# Prepares data for training the grey matter segmentation model
source "${PROJECT_PATH}/scripts/data_preparation/prepare_data.sh"

##################
# Segmentation #
##################

# Trains the grey matter segmentation model
python -u scripts/segmentation/train.py

# ################
# # Localisation #
# ################

# Segments grey matter and locates the plaques
SEGMENTATION_MODEL='models/segmentation/unet/22023-03-15-unet-16x-bs16-ps256-lr0.0001-e3v49.pt'
python -u scripts/localisation/locate.py -sf "${SOURCE_FOLDER1}" -tf "${TARGET_FOLDER}" -mp ${SEGMENTATION_MODEL} -s0 ${START_IDX} -s1 ${STOP_IDX}
python -u scripts/localisation/locate.py -sf "${SOURCE_FOLDER2}" -tf "${TARGET_FOLDER}" -mp ${SEGMENTATION_MODEL} -s0 ${START_IDX} -s1 ${STOP_IDX}


##################
# Categorisation #
##################

# Finetunes models using labeled data
python -u scripts/categorisation/supervised/finetune.py -c configs/supervised.yaml

# Generates thresholds for finetuned models
python -u scripts/categorisation/supervised/evaluate.py -c configs/supervised.yaml

# Assigns classes to plaques based on ensemble model
python -u scripts/categorisation/supervised/predict.py -c configs/supervised.yaml

# Visualises class predictions
N_IMAGES=20
IMG_SIZE=128
PRED_PATH='result/class_assignment/resnet50_moco.npz'
python -u scripts/categorisation/class_report.py -noi "${N_IMAGES}" -ip "${TARGET_FOLDER}" -ap "${PRED_PATH}"
python -u scripts/categorisation/visualisation.py -sup -ipc "${N_IMAGES}" -is "${IMG_SIZE}" -ap "${PRED_PATH}"


##################
#     Plots      #
##################

# Computes AB loads
python -u scripts/categorisation/compute_load.py -c configs/plots.yaml

# Generates frequency count bar plots
python -u scripts/categorisation/stats/freq_bar_plots.py -c configs/plots.yaml

# Generates AB load box/violin plots
python -u scripts/categorisation/stats/load_violin_plots.py -c configs/plots.yaml

# Generates participant data correlation matrices
python -u scripts/categorisation/stats/correlation_matrices.py -c configs/plots.yaml

# Generates staging scheme scatter plots
python -u scripts/categorisation/stats/phase_scatter_plots.py -c configs/plots.yaml