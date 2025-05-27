import numpy as np

from display_HDOutliers import display_HDOutliers
from find_HDOutliers import find_HDOutliers


# Converted from the following R code:
# library(stray)
# require(ggplot2)
# #> Loading required package: ggplot2
# set.seed(1234)
# data <- c(rnorm(1000, mean = -6), 0, rnorm(1000, mean = 6))
# outliers <- find_HDoutliers(data, knnsearchtype = "brute")
# names(outliers)
# #> [1] "outliers"   "out_scores" "type"
# display_HDoutliers(data, outliers)

def main():
    # Set random seed for reproducibility
    np.random.seed(1234)
    # Generate 1000 samples from a normal distribution with mean -6
    data = np.concatenate([
        np.random.normal(-6, 1, 1000),
        [0],  # Add outlier
        np.random.normal(6, 1, 1000)
    ])
    # Reshape data to 2D array with one column (required format)
    data = data.reshape(-1, 1)
    # Detect outliers
    results = find_HDOutliers(data, knn_search_type='brute')
    # Visualize results
    display_HDOutliers(data, results)


if __name__ == "__main__":
    main()
