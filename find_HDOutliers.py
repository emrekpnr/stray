from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

from find_threshold import find_threshold
from normalize_data import normalize_data


def use_KNN(data: np.ndarray, alpha: float, k: int, knn_search_type: str, p: float, tn: int):
    # Check if the data is empty and raise an error if so
    if data.shape[0] == 0:
        raise ValueError("Data cannot be empty")
    # Fit a KNN model using the provided data and search type
    nearest_neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm=knn_search_type).fit(data)
    # Get the distances of the k nearest neighbors for each point, excluding the self-distance
    nearest_neighbor_distances = nearest_neighbors.kneighbors(data, return_distance=True)[0][:, 1:]
    # Add a column of zeros to represent the self-distance (which is always zero)
    all_neighbor_distances = np.hstack([np.zeros((nearest_neighbor_distances.shape[0], 1)),
                                        nearest_neighbor_distances])
    # Calculate the difference in distances between consecutive nearest neighbors
    differences = np.diff(all_neighbor_distances, axis=1)
    # Identify the index of the largest difference (gap) in distances for each point
    max_distance_diff_indices = np.argmax(differences, axis=1)
    # Get the maximum gap distances corresponding to the largest difference for each point
    max_gap_distances = nearest_neighbor_distances[
        np.arange(len(nearest_neighbor_distances)), max_distance_diff_indices]
    # Use the find_threshold method to compute the outlier indices based on the max gap distances
    outlier_indices = find_threshold(max_gap_distances, alpha=alpha, out_tail="max", p=p, tn=tn)

    return {"outliers": outlier_indices, "out_scores": max_gap_distances}


def find_HDOutliers(data: Union[np.ndarray], alpha: float = 0.01, k: int = 10, knn_search_type: str = "brute",
                    normalize: str = "unitize", p: float = 0.5, tn: int = 50):
    # Validate input parameters
    data = np.asarray(data)
    r = data.shape[0]

    # Mask missing values and clean data
    mask = np.all(np.isfinite(data), axis=1)
    clean_data = data[mask]

    # If no valid data remains after cleaning, raise an error
    if clean_data.shape[0] == 0:
        raise ValueError("No valid data after removing rows with missing values")

    # Generate tags for the cleaned data (indices of the valid rows in the original data)
    tag = np.arange(r)[mask]
    # Normalize the cleaned data based on the specified method
    normalized_data = normalize_data(clean_data, method=normalize)
    # Use KNN to calculate outlier scores and find outliers
    out_items = use_KNN(normalized_data, alpha=alpha, k=k, knn_search_type=knn_search_type, p=p, tn=tn)
    # Get the indices of the detected outliers
    outliers = tag[out_items["outliers"]]
    # Classify all points as 'typical' by default, then mark the outliers as 'outlier'
    types = np.array(["typical"] * r)
    types[outliers] = "outlier"

    return {"outliers": outliers, "out_scores": out_items["out_scores"], "type": types}
