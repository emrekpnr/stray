import unittest
import numpy as np

from normalize_data import normalize_data


class TestNormalizeData(unittest.TestCase):

    def test_unitize_method(self):
        data = np.array([[1, 2, 3], [4, 7, 6], [7, 8, 9]])
        normalized_data = normalize_data(data, method="unitize")
        # Check that the values are between 0 and 1 for each column
        self.assertTrue(np.all(normalized_data >= 0) and np.all(normalized_data <= 1))
        # Check that the values are equal to [[0.0, 0.0, 0.0],[0.5, 0.8333, 0.5],[1.0, 1.0, 1.0]]
        expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.8333, 0.5], [1.0, 1.0, 1.0]])
        self.assertTrue(np.allclose(normalized_data, expected, rtol=1e-4))

    def test_standardize_method(self):
        data = np.array([[1, 2, 3], [4, 7, 6], [7, 8, 9]])
        normalized_data = normalize_data(data, method="standardize")
        # Check that the values are standardized (mean=0, IQR=1 for each column)
        for i in range(data.shape[1]):
            column = normalized_data[:, i]
            self.assertAlmostEqual(np.median(column), 0, places=1)
            self.assertAlmostEqual(np.percentile(column, 75) - np.percentile(column, 25), 1, places=1)
        # Check that the values are equal to [[-1.0, -1.6667, -1.0 ],[0.0, 0.0, 0.0],[1.0, 0.3333, 1.0]]
        expected = np.array([[-1.0, -1.6667, -1.0], [0.0, 0.0, 0.0], [1.0, 0.3333, 1.0]])
        self.assertTrue(np.allclose(normalized_data, expected, rtol=1e-4))

    def test_identical_values(self):
        data = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
        normalized_data = normalize_data(data, method="unitize")
        # Check that all values are zeros when all values in the column are identical
        self.assertTrue(np.all(normalized_data == 0))

    def test_single_data_point(self):
        data = np.array([[3]])
        normalized_data = normalize_data(data, method="unitize")
        # Single data point should return a single zero in unitize normalization
        self.assertTrue(np.all(normalized_data == 0))

if __name__ == '__main__':
    unittest.main()
