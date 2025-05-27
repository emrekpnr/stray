import numpy as np

from display_HDOutliers import display_HDOutliers
from find_HDOutliers import find_HDOutliers


# Converted from the following R code:
# set.seed(1234)
# n <- 1000 # number of observations
# nout <- 10 # number of outliers
# typical_data <- matrix(rnorm(2*n), ncol = 2, byrow = TRUE)
# out <- matrix(5*runif(2*nout,min=-5,max=5), ncol = 2, byrow = TRUE)
# data <- rbind(out, typical_data )
# outliers <- find_HDoutliers(data, knnsearchtype = "brute")
# display_HDoutliers(data, outliers)


def main():
    # Set random seed for reproducibility
    np.random.seed(1234)

    number_of_typical_observations = 1000
    number_of_outliers = 10

    # Typical data: 2D standard normal
    typical_data = np.random.randn(number_of_typical_observations, 2)
    # Outliers: uniformly scattered in a larger range
    outliers_data = 5 * np.random.uniform(-5, 5, size=(number_of_outliers, 2))
    # Combine outliers and typical data
    data = np.vstack([outliers_data, typical_data])
    # Run STRAY algorithm (your Python version)
    results = find_HDOutliers(data, knn_search_type='brute')
    # Visualize
    display_HDOutliers(data, results)


if __name__ == "__main__":
    main()
