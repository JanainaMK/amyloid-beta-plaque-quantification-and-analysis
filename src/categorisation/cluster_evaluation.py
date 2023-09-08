import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def scores(features, assignments, metric):
    sil = silhouette_score(features, assignments, metric=metric)
    ch = calinski_harabasz_score(features, assignments)
    if metric == 'euclidean':
        db = davies_bouldin_score(features, assignments)


