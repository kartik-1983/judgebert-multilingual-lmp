"""Unit tests for robust PreserveLex regression metrics.

Run from judgebert_multilingual/: python -m unittest discover -s tests
"""
import unittest

import numpy as np

from src.utils import compute_metrics


class ComputeMetricsTests(unittest.TestCase):
    def test_computes_regression_metrics(self):
        result = compute_metrics(np.array([1, 2, 3]), np.array([1, 2, 2]))
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mae"], 1 / 3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(1 / 3))

    def test_constant_vector_returns_nan_correlations(self):
        result = compute_metrics([2, 2, 2], [1, 2, 3])
        self.assertTrue(np.isnan(result["pearson"]))
        self.assertTrue(np.isnan(result["spearman"]))
        self.assertEqual(result["n"], 3)

    def test_rejects_single_observation(self):
        with self.assertRaises(ValueError):
            compute_metrics([1], [1])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            compute_metrics([1, 2], [1])

    def test_rejects_non_finite_values(self):
        with self.assertRaises(ValueError):
            compute_metrics([1, np.nan], [1, 2])


if __name__ == "__main__":
    unittest.main()
