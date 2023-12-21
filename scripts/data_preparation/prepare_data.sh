#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/path/to/project"

python -u scripts/data_preparation/downsample_images.py
python -u scripts/data_preparation/downsample_labels.py 
python -u scripts/data_preparation/link_images_labels.py --split_data