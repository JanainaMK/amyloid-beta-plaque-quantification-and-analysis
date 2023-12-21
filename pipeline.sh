#!/bin/bash

export PROJECT_PATH="/path/to/project"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"

# start and stop image indices 
START_IDX=0
STOP_IDX=10

CLUSTERS=10
SOURCE_FOLDER='dataset/original/Abeta images 100+'
TARGET_FOLDER='result/features'

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

################
# Localisation #
################

# Segments grey matter and locates the plaques
python -u scripts/localisation/locate.py -sf "${SOURCE_FOLDER}" -tf "${TARGET_FOLDER}" -s0 ${START_IDX} -s1 ${STOP_IDX}


##################
# Categorisation #
##################

# Generates AlexNet features based on located plaques' bounding boxes
python -u scripts/categorisation/features.py -s0 ${START_IDX} -s1 ${STOP_IDX}

python -u scripts/categorisation/cluster.py --alex_features --simple_features --fit --predict -noc ${CLUSTERS} -s0 ${START_IDX} -s1 ${STOP_IDX}