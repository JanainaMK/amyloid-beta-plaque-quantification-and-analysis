import os
import sys
import subprocess
import argparse

import src.util.cli as cli

print('starting')
parser = argparse.ArgumentParser()
cli.add_io_settings(parser)
cli.add_segmentation_settings(parser)
cli.add_series_settings(parser)

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
    subprocess.run([sys.executable,  '-u',  'script/segmentation/segment.py', name[:-4],
                    '--source_folder', str(args.source_folder),
                    '--target_folder', str(args.target_folder),
                    '--patch_size_segmentation', str(args.patch_size_segmentation),
                    '--downscale_factor', str(args.downscale_factor),
                    '--model_path', str(args.model_path),
                    ])
    print()
print('series finished')