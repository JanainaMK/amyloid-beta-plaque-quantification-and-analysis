import subprocess
import argparse

import h5py

print('starting')
parser = argparse.ArgumentParser()
parser.add_argument('-s0', '--start', default=0, type=int)
parser.add_argument('-s1', '--stop', default=579, type=int)
parser.add_argument('-c', '--cases', default='all', type=str)
args = parser.parse_args()
print(args)


start = args.start
stop = args.stop
cases = args.cases

entry_file_path = 'dataset/AD+cent.hdf5'
entry_file = h5py.File(entry_file_path, 'r')


names = []
for entry_name in entry_file:
    if entry_file[f'{entry_name}/image_file'].attrs['case'] == cases or cases == 'all':
        names.append(entry_name)

print(len(names), 'images found')
print('processing', stop - start, f'slides ({start} - {stop})')

for i, name in enumerate(names):
    if i < start or i >= stop:
        continue
    print('slide:', i)
    subprocess.run(['python',  '-u',  'script/localisation/loc.py', name, '-uo'])
print('series finished')

