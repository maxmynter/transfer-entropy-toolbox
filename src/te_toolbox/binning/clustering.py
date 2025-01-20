"""Clustering based discretization methods."""

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift
from sklearn.neighbors import NearestNeighbors


def kmeans_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate bin edges using k-means clustering.

    Number of clusters is determined by square root of data length.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of bin edges based on cluster boundaries

    """
    n_clusters = int(np.sqrt(len(data)))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data.reshape(-1, 1))

    centers = np.sort(kmeans.cluster_centers_.flatten())
    edges = np.zeros(len(centers) + 1)
    edges[0] = np.min(data)
    edges[-1] = np.max(data)
    edges[1:-1] = (centers[1:] + centers[:-1]) / 2

    return edges


def get_optimal_eps(data: npt.NDArray[np.float64], max_neighbors: int = 10) -> float:
    """Determine optimal epsilon for DBSCAN using nearest neighbor distances.

    Args:
    ----
        data: Input data array
        max_neighbors: Maximum number of neighbors to consider

    Returns:
    -------
        Optimal epsilon value

    """
    nbrs = NearestNeighbors(n_neighbors=max_neighbors + 1).fit(data.reshape(-1, 1))
    distances, _ = nbrs.kneighbors(data.reshape(-1, 1))

    mean_dists = np.mean(distances[:, 1:], axis=0)
    diffs = np.diff(mean_dists)
    optimal_k = np.argmax(diffs)

    return mean_dists[optimal_k]


def dbscan_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate bin edges using DBSCAN clustering.

    Epsilon is determined automatically using nearest neighbor distances.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of bin edges based on cluster boundaries

    """
    eps = get_optimal_eps(data)
    min_samples = max(5, int(0.01 * len(data)))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data.reshape(-1, 1))

    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        unique_labels = unique_labels[1:]

    edges = [np.min(data)]
    for label in unique_labels:
        cluster_data = data[labels == label]
        edges.append(np.min(cluster_data))
    edges.append(np.max(data))

    return np.unique(edges)


def meanshift_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate bin edges using Mean Shift clustering.

    Bandwidth is estimated automatically.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of bin edges based on cluster boundaries

    """
    ms = MeanShift(bandwidth=None)
    labels = ms.fit_predict(data.reshape(-1, 1))

    unique_labels = np.unique(labels)
    edges = [np.min(data)]
    for label in unique_labels:
        cluster_data = data[labels == label]
        edges.append(np.min(cluster_data))
    edges.append(np.max(data))

    return np.unique(edges)


def agglomerative_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate bin edges using Agglomerative clustering.

    Number of clusters is determined by square root of data length.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of bin edges based on cluster boundaries

    """
    n_clusters = int(np.sqrt(len(data)))
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(data.reshape(-1, 1))

    unique_labels = np.unique(labels)
    edges = [np.min(data)]
    for label in unique_labels:
        cluster_data = data[labels == label]
        edges.append(np.min(cluster_data))
    edges.append(np.max(data))

    return np.unique(edges)
