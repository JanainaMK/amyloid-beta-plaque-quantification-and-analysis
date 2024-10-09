#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
DOWNSAMPLE_LEVEL=16
SAVE_DIR='dataset'

python -u scripts/data_preparation/downsample_images.py --save_dir "${SAVE_DIR}" --vsi_dir 'dataset/Abeta images 100+' -dl "${DOWNSAMPLE_LEVEL}"
python -u scripts/data_preparation/downsample_images.py --save_dir "${SAVE_DIR}" --vsi_dir 'dataset/Abeta images AD cohort' -dl "${DOWNSAMPLE_LEVEL}"

python -u scripts/data_preparation/downsample_labels.py --save_dir "${SAVE_DIR}" -ld 'dataset/grey matter labels' -dl "${DOWNSAMPLE_LEVEL}"
python -u scripts/data_preparation/link_images_labels.py --split_data --save_dir "${SAVE_DIR}" 