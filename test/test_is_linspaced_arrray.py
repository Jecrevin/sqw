import unittest

import numpy as np

from app.math import is_linspaced_array


class TestIsLinspacedArray(unittest.TestCase):
    def test_raises_for_non_1d_array(self):
        """Test if ValueError is raised for non-1D arrays."""
        arr_2d = np.array([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "Input array must be 1-dimensional"):
            is_linspaced_array(arr_2d)

    def test_linspaced_arrays(self):
        """Test if the function correctly identifies various linearly spaced arrays."""
        self.assertTrue(is_linspaced_array(np.array([1, 2, 3, 4, 5])))
        self.assertTrue(is_linspaced_array(np.array([0.1, 0.2, 0.3, 0.4])))
        self.assertTrue(is_linspaced_array(np.linspace(0, 100, 50)))
        self.assertTrue(is_linspaced_array(np.array([5, 4, 3, 2, 1])))  # Decreasing sequence
        self.assertTrue(is_linspaced_array(np.array([-1, -2, -3, -4])))  # Negative number sequence
        self.assertTrue(is_linspaced_array(np.array([0, 1e-16, 2e-16, 3e-16])))  # Small step size

    def test_not_linspaced_arrays(self):
        """Test if the function correctly identifies various non-linearly spaced arrays."""
        self.assertFalse(is_linspaced_array(np.array([1, 2, 4, 5])))
        self.assertFalse(is_linspaced_array(np.array([0.1, 0.2, 0.25, 0.4])))
        self.assertFalse(is_linspaced_array(np.array([0, 1e-16, 3e-16])))
        # small step size non-uniform array
        self.assertFalse(is_linspaced_array(np.array([0, 1e-16, 2.1e-16, 3.1e-16])))

    def test_edge_cases(self):
        """Test edge cases: empty, single-element, and two-element arrays."""

        self.assertTrue(is_linspaced_array(np.array([])))
        self.assertTrue(is_linspaced_array(np.array([10])))

        # Two-element arrays are always linearly spaced
        self.assertTrue(is_linspaced_array(np.array([10, 20])))
