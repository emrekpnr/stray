import time

import numpy as np

from find_HDOutliers import find_HDOutliers


def estimate_fpr(knn_search_type, dim, n_samples=1000, n_iterations=100):
    false_positive_rates = []

    for _ in range(n_iterations):
        # Generate data with no outliers
        data = np.random.randn(n_samples, dim)
        result = find_HDOutliers(data, knn_search_type=knn_search_type)
        num_outliers = sum(1 for t in result["type"] if t == "outlier")
        fpr = num_outliers / n_samples
        false_positive_rates.append(fpr)

    return np.mean(false_positive_rates)


def main():
    dims = [1, 10, 100]
    sample_sizes = [100, 500, 1000, 2500, 5000, 7500, 10000]
    knn_types = ["brute", "kd_tree", "ball_tree", "auto"]

    for knn_type in knn_types:
        print(f"\n=== Results for KNN Search Type: {knn_type} ===")
        for dim in dims:
            row = f"dim {dim}: "
            for n in sample_sizes:
                start_time = time.time()
                fpr = estimate_fpr(knn_type, dim, n_samples=n)
                duration = time.time() - start_time
                row += f"{fpr:.3f} ({duration:.2f}s)  "
            print(row)


if __name__ == "__main__":
    main()
