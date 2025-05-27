import unittest

import numpy as np

from find_threshold import find_threshold


class TestFindThreshold(unittest.TestCase):
    def test_detects_known_anomalies(self):
        # Arrange: create mock scores with clear anomalies
        np.random.seed(42)
        normal_scores = np.random.normal(0, 1, 100)
        anomalies = np.array([6.0, 6.5, 7.0])
        scores = np.concatenate([normal_scores, anomalies])

        # Act: run the detection
        detected = find_threshold(scores)

        # Assert: check that known anomalies are found
        expected_anomaly_indices = [100, 101, 102]
        for idx in expected_anomaly_indices:
            self.assertIn(idx, detected, f"Anomaly at index {idx} was not detected")


if __name__ == '__main__':
    unittest.main()
