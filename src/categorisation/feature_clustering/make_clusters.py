import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering


def hierarchical_clustering(
    features: np.ndarray,
    n_clusters: int = None,
    distance_threshold: float = None,
    linkage="single",
    metric="cosine",
):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage,
        metric=metric,
        compute_distances=True,
    )
    assignments = model.fit_predict(features)
    return model, assignments


def dbscan_clustering(
    features: np.ndarray,
    radius: float,
    min_samples: int,
    metric="cosine",
):
    model = DBSCAN(
        eps=radius,
        min_samples=min_samples,
        metric=metric,
        n_jobs=-1,
        algorithm="kd_tree",
    )
    assignments = model.fit_predict(features)
    return model, assignments
