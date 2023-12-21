import argparse
import os
import time
import sys

import h5py
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import joblib


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='kmeans', type=str)
parser.add_argument('-dm', '--distance_metric', default='cosine', type=str)
parser.add_argument('-noc', '--number_of_clusters', default=20, type=int)
parser.add_argument('-af', '--alex_features', action='store_true')
parser.add_argument('-sf', '--simple_features', action='store_true')
parser.add_argument('-f', '--fit', action='store_true', default=False)
parser.add_argument('-p', '--predict', action='store_true', default=False)
parser.add_argument('-s0', '--start', default=0, type=int)
parser.add_argument('-s1', '--stop', default=579, type=int)
args = parser.parse_args()
print(args)

model_string = args.model
distance_metric = args.distance_metric
n_clusters = args.number_of_clusters
alex_features = args.alex_features
simple_features = args.simple_features
start = time.time()

root = 'result/features'

print('collecting features...')
m = 0
if alex_features:
    m += 1000
    feature_string = 'alex'
if simple_features:
    m += 2
    feature_string = 'simple'
if alex_features and simple_features:
    feature_string = 'alex+simple'
if m == 0:
    raise ValueError('please select features')

features = np.zeros((0, m))
for i, filename in enumerate(os.listdir(root)[args.start:args.stop]):
    with h5py.File(os.path.join(root, filename), 'r') as file:
        # concatenates features
        local = np.zeros((file.attrs['n_plaques'], 0))
        if alex_features: 
            local = np.concatenate((local, file['alex_features'][()]), 1)
        if simple_features:
            area = np.expand_dims(file['area'][()], 1)
            roundness = np.expand_dims(file['roundness'][()], 1)
            local = np.concatenate((local, area, roundness), 1)
        # concatenates plaque samples
        features = np.concatenate((features, local), 0)
        print(filename, 'collected', time.time() - start)
print('features concatenated, total:', features.shape[0])

name_string = f'{model_string}-dm{distance_metric}-nc{n_clusters}-{feature_string}'
# fit the model based on plaque features
if args.fit:
    if model_string == 'kmeans':
        model = KMeans(n_clusters=n_clusters, n_init='auto', init='k-means++')
    elif model_string == 'mbkmeans':
        model = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto', init='k-means++')
    else:
        print(model_string, 'is not a valid model')
        sys.exit()
    print(model_string, 'model selected', time.time() - start)
    
    assignments = model.fit(features)
    joblib.dump(model, f'result/cluster-model/{name_string}.joblib')
    print('model saved', time.time() - start)
# generate predictions based on plaque features
if args.predict:
    model = joblib.load(f'result/cluster-model/{name_string}.joblib')
    print('model loaded')
    assignments = model.predict(features)
    n_clusters = assignments.max() + 1
    print('clusters found:', n_clusters, time.time() - start)

    np.save(f'result/cluster-assignment/{name_string}.npy', assignments)
    print('assignments saved', time.time() - start)
