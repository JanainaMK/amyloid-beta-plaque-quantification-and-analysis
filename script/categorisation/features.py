import os
import sys
import time
import argparse

import numpy as np
import torch
import h5py

from src.categorisation.pre_processing import rgb_to_dab, alex_prep
from src.categorisation.extract_features import generate_alex_features

parser = argparse.ArgumentParser()
parser.add_argument('-start', '--start_at', default=0, type=int)
parser.add_argument('-stop', '--stop_at', default=579, type=int)
parser.add_argument('-c', '--cases', default='all', type=str)
args = parser.parse_args()

start_at = args.start_at
stop_at = args.stop_at
cases = args.cases

if not torch.cuda.is_available():
    print('cuda not availible, failing')
    sys.exit()


print('starting...')

root = 'result/images'

names = []

for filename in os.listdir(root):
    try:
        file = h5py.File(os.path.join(root, filename), 'r')
        if file.attrs['case'] == cases or cases == 'all':
            names.append(filename)
    except BlockingIOError:
        print(filename, 'is already open, skipping.')
        continue
    except KeyError:
        print(filename, 'has no case attribute')
        continue
    file.close()
print(len(names), 'slides found')

for j, filename in enumerate(names):
    start = time.time()
    if j < start_at:
        continue
    elif j >= stop_at:
        break

    try:
        file = h5py.File(os.path.join(root, filename), 'a')
        n = file['plaques'].attrs['length']
        n_act = 0
    except BlockingIOError:
        print('file', filename, 'already open, failing feature collection')
        continue
    except KeyError:
        print('file', filename, 'missing length attribute, failing feature collection')
        continue

    fail = False
    print(filename, '-', n, 'plaques:')
    features = np.zeros((n, 1000))
    for i in range(n):
        try:
            plaque = file[f'plaques/{i}']
        except KeyError:
            fail = True
            continue
        n_act += 1
        image = plaque['plaque'][()]
        pre = rgb_to_dab(image)
        tensor = alex_prep(pre)
        feature = generate_alex_features(tensor).detach().cpu().numpy()
        features[i] = feature
        if 'alex_feature' in plaque:
            try:
                del plaque['alex_feature']
            except KeyError as e:
                print(e)
                print('plaque deletion failed, exiting')
                break
        plaque.create_dataset('alex_feature', data=feature)
    if fail:
        print(f'{n_act}/{n} plaques found, feature extraction failed')
        continue
    if 'alex_features' in file:
        del file['alex_features']
    file.create_dataset('alex_features', data=features)
    file.close()
    print('Alex features saved', time.time() - start)
