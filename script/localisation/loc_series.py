import os
import sys
import subprocess
import argparse

import src.util.cli as cli

print('starting')
parser = argparse.ArgumentParser(prog='plaque localisation - series', description='An algorithm that can find Ab plaques in multiple vsi file')
cli.add_io_settings(parser)
cli.add_segmentation_settings(parser)
cli.add_localisation_settings(parser)
cli.add_series_settings(parser)
parser.add_argument('-ss', '--segmentation_setting', choices=['no', 'load', 'create'], default='create', help='The way the grey matter segmentation is handled. \n \'no\' skips the segmentation \n \'load\' loads the segmentation from the result file located in the target_folder \n \'create\' creates a new segmentation and stores it in the target_folder.')
args = parser.parse_args()
print(args)


start = args.start
stop = args.stop
vsi_folder = args.source_folder


file_names = os.listdir(vsi_folder)
print(len(file_names), 'files found in total')
names = file_names[start:stop]

print('processing', stop - start, f'slides ({start} - {stop})')

for i, name in enumerate(names):
    print('slide', i, ':', name)
    subprocess.run([sys.executable,  '-u',  'script/localisation/loc.py', name[:-4],
                    '--source_folder', str(args.source_folder),
                    '--target_folder', str(args.target_folder),
                    '--patch_size_segmentation', str(args.patch_size_segmentation),
                    '--downscale_factor', str(args.downscale_factor),
                    '--model_path', str(args.model_path),
                    '--patch_size_localisation', str(args.patch_size_localisation),
                    '--threshold', str(args.threshold),
                    '--kernel_size', str(args.kernel_size),
                    '--minimum_size', str(args.minimum_size),
                    '--segmentation_setting', str(args.segmentation_setting),
                    ])
    print()
print('series finished')

