import numpy as np


def find_threshold(outlier_score, alpha=0.01, out_tail="max", p=0.5, tn=50):
    # Convert the input to a NumPy array
    outlier_score = np.array(outlier_score)
    n = len(outlier_score)

    # Invert the scores if we're detecting minimum outliers (i.e., lower values are more "outlier-like")
    if out_tail == "min":
        outlier_score = -outlier_score
    elif out_tail != "max":
        raise ValueError("out_tail must be either 'max' or 'min'")
    # Sort the outlier scores in ascending order and get the sorted indices
    sorted_score_indices = np.argsort(outlier_score)
    sorted_scores = outlier_score[sorted_score_indices]
    # Calculate the differences between consecutive sorted scores
    score_differences = np.diff(sorted_scores, prepend=0)
    # Set the maximum sample threshold (based on n and tn)
    max_sample_threshold = max(min(tn, n // 4), 2)
    # Define the range of gap samples to consider for the calculation
    gap_sample_range = np.arange(2, max_sample_threshold + 1)
    # Start the search for the threshold from a specific index based on the proportion p
    start_index_for_search = max(int(np.floor(n * (1 - p))), 1)

    # Initialize an array to store the gap sum estimate for each point
    gap_sum_estimate = np.zeros(n)
    # Estimate the gap sum for each point starting from the computed search index
    for i in range(start_index_for_search, n):
        indices = i - gap_sample_range + 1  # Get the indices for the current gap sample
        indices = indices[indices >= 0]  # Avoid negative indices
        gap_sum_estimate[i] = np.sum(
            (gap_sample_range[:len(indices)] / (max_sample_threshold - 1)) * score_differences[indices])
    # Calculate the log of the inverse alpha value for the threshold
    log_alpha_threshold = np.log(1 / alpha)
    bound = np.inf
    # Iterate through the scores and compare each score's difference to the threshold
    for i in range(start_index_for_search, n):
        if score_differences[i] > log_alpha_threshold * gap_sum_estimate[i]:
            bound = sorted_scores[i - 1]  # Set the bound based on the sorted scores
            break

    # Find the indices of the outliers based on the computed threshold
    outlier_indices = np.where(outlier_score > bound)[0]
    return outlier_indices.tolist()

