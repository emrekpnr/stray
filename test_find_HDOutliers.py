import unittest

import numpy as np

from find_HDOutliers import find_HDoutliers, use_KNN


class TestFindHDOutliers(unittest.TestCase):

    def test_use_knn_outliers(self):
        # Test for outlier detection using KNN
        data = np.array([[1, 2], [2, 3], [3, 4], [10, 10]])  # 10, 10 should be an outlier
        result = use_KNN(data, alpha=0.01, k=2, knn_search_type="brute", p=0.5, tn=50)
        self.assertIn(3, result["outliers"])  # Index 3 (the outlier) should be detected

    def test_use_knn_empty_data(self):
        # Test for empty data
        data = np.array([]).reshape(0, 2)
        with self.assertRaises(ValueError):
            use_KNN(data, alpha=0.01, k=2, knn_search_type="brute", p=0.5, tn=50)

    def test_use_knn_invalid_knn_search_type(self):
        # Test for invalid knn search type
        data = np.array([[1, 2], [2, 3], [3, 4], [10, 10]])
        with self.assertRaises(ValueError):
            use_KNN(data, alpha=0.01, k=2, knn_search_type="invalid", p=0.5, tn=50)

    def test_find_hd_outliers_basic(self):
        # Test for basic outlier detection
        data = np.array([[1, 2], [2, 3], [3, 4], [10, 10]])  # 10, 10 should be an outlier
        result = find_HDoutliers(data, alpha=0.01, k=2, knn_search_type="brute", normalize="unitize", p=0.5, tn=50)
        self.assertIn(3, result["outliers"])  # Index 3 should be detected as an outlier
        self.assertEqual(len(result["out_scores"]), len(data))  # Check that scores correspond to the data size

    def test_find_hdoutliers_missing_values(self):
        # Test for handling missing values (NaN or inf)
        data = np.array([[1, 2], [np.nan, 3], [3, 4], [10, np.inf]])  # NaN and inf values
        with self.assertRaises(ValueError):
            find_HDoutliers(data, alpha=0.01, k=2, knn_search_type="brute", normalize="unitize", p=0.5, tn=50)

    def test_find_hdoutliers_identical_data(self):
        # Test for edge case with identical data (no outliers)
        data = np.array([[5, 5], [5, 5], [5, 5], [5, 5]])
        result = find_HDoutliers(data, alpha=0.01, k=2, knn_search_type="brute", normalize="unitize", p=0.5, tn=50)
        self.assertEqual(len(result["outliers"]), 0)  # No outliers should be detected


if __name__ == "__main__":
    unittest.main()
