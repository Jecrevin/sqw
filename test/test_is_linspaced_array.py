import unittest
import numpy as np
from app.math import is_linspaced_array


class TestIsLinspacedArray(unittest.TestCase):
    def test_non_1d_array_raises_error(self):
        """Test that non-1D arrays raise ValueError"""
        with self.assertRaisesRegex(ValueError, "Input array must be one-dimensional"):
            is_linspaced_array(np.array([[1, 2], [3, 4]]))

        with self.assertRaisesRegex(ValueError, "Input array must be one-dimensional"):
            is_linspaced_array(np.array([[[1]]]))

    def test_empty_array(self):
        """Test empty array returns True"""
        self.assertTrue(is_linspaced_array(np.array([])))

    def test_single_element(self):
        """Test single element array returns True"""
        self.assertTrue(is_linspaced_array(np.array([5])))

    def test_two_elements(self):
        """Test two element array returns True"""
        self.assertTrue(is_linspaced_array(np.array([1, 3])))
        self.assertTrue(is_linspaced_array(np.array([10, 10])))

    def test_linearly_spaced_arrays(self):
        """Test various linearly spaced arrays return True"""
        # Standard linear spacing
        self.assertTrue(is_linspaced_array(np.array([1, 2, 3, 4, 5])))
        self.assertTrue(is_linspaced_array(np.array([0, 0.5, 1.0, 1.5, 2.0])))

        # Negative spacing
        self.assertTrue(is_linspaced_array(np.array([5, 4, 3, 2, 1])))

        # Zero spacing (constant array)
        self.assertTrue(is_linspaced_array(np.array([7, 7, 7, 7])))

        # Using np.linspace
        self.assertTrue(is_linspaced_array(np.linspace(0, 10, 11)))
        self.assertTrue(is_linspaced_array(np.linspace(-5, 5, 21)))

    def test_non_linearly_spaced_arrays(self):
        """Test non-linearly spaced arrays return False"""
        self.assertFalse(is_linspaced_array(np.array([1, 2, 4, 8])))
        self.assertFalse(is_linspaced_array(np.array([1, 3, 5, 8])))
        self.assertFalse(is_linspaced_array(np.array([0, 1, 1, 2])))

    def test_small_differences_high_precision(self):
        """Test arrays with very small differences"""
        # Small differences should use high precision (1e-16)
        small_diff_array = np.array([0, 1e-16, 2e-16, 3e-16])
        self.assertTrue(is_linspaced_array(small_diff_array))

        # Slightly non-uniform small differences
        non_uniform_small = np.array([0, 1e-16, 2.1e-16, 3e-16])
        self.assertFalse(is_linspaced_array(non_uniform_small))

    def test_large_differences_standard_tolerance(self):
        """Test arrays with larger differences using standard tolerance"""
        # Large differences should use atol=1e-8
        large_diff_array = np.array([0, 1, 2, 3])
        self.assertTrue(is_linspaced_array(large_diff_array))

        # Within tolerance
        slightly_off = np.array([0, 1, 2.0000001, 3])
        self.assertTrue(is_linspaced_array(slightly_off))

        # Outside tolerance
        too_far_off = np.array([0, 1, 2.001, 3])
        self.assertFalse(is_linspaced_array(too_far_off))

    def test_floating_point_precision(self):
        """Test handling of floating point precision issues"""
        # Array that might have small floating point errors
        arr = np.array([0.1 * i for i in range(10)])
        self.assertTrue(is_linspaced_array(arr))

        # Array with accumulated floating point errors
        arr_with_errors = np.array([0.1, 0.2, 0.30000000000000004, 0.4])
        self.assertTrue(is_linspaced_array(arr_with_errors))


if __name__ == "__main__":
    unittest.main()
