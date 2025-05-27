import numpy as np

from display_HDOutliers import display_HDOutliers
from find_HDOutliers import find_HDOutliers


# Converted from the following R code:
# data <- rbind(matrix(rnorm(144), ncol = 3), c(10,12,10),c(3,7,10))
# output <- find_HDoutliers(data, knnsearchtype = "brute")
# display_HDoutliers(data, out = output)


def main():
    # Set random seed for reproducibility
    # Generate 48 rows (144 elements reshaped into 48x3 matrix)
    typical_data = np.random.randn(48, 3)

    # Add two 3D outliers
    outliers = np.array([[10, 12, 10], [3, 7, 10]])
    # Combine typical data and outliers
    data = np.vstack([typical_data, outliers])
    # Run STRAY detection
    results = find_HDOutliers(data, knn_search_type="brute")
    # Display result
    display_HDOutliers(data, results)


if __name__ == "__main__":
    main()
