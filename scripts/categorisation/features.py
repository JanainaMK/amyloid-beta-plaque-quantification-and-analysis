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
parser.add_argument('-s0', '--start', default=0, type=int)
parser.add_argument('-s1', '--stop', default=579, type=int)
args = parser.parse_args()

start_at = args.start
stop_at = args.stop

if not torch.cuda.is_available():
    print('cuda not availible, failing')
    sys.exit()

print('starting...')

root = 'result/features'

names = os.listdir(root)

print(len(names), 'slides found')

for j, filename in enumerate(names):
    start = time.time()
    if j < start_at:
        continue
    elif j >= stop_at:
        break
    
    file = h5py.File(os.path.join(root, filename), 'a')
    try:
        n = file['plaques'].attrs['length']
        n_act = 0

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
    except BlockingIOError:
        print('file', filename, 'already open, failing feature collection')
        continue
    except KeyError:
        print('file', filename, 'missing length attribute, failing feature collection')
        continue
    finally:
        file.close()
    print('Alex features saved', time.time() - start)
