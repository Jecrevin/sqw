import unittest
import numpy as np
from app.math import arr_mul_dft


class TestFuncMulDFT(unittest.TestCase):
    """Test suite for the func_mul_dft function."""

    def setUp(self):
        """Set up test data."""
        self.arr1_same = np.array([1, 2, 3, 4])
        self.arr2_same = np.array([5, 6, 7, 8])
        self.arr1_diff = np.array([1, 2, 3, 4, 5])
        self.arr2_diff = np.array([1, 2, 3])

    def test_raises_value_error_for_non_1d_arrays(self):
        """Test that func_mul_dft raises ValueError for non-1D inputs."""
        a_1d = np.array([1, 2, 3])
        a_2d = np.array([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "Both input arrays must be 1-dimensional."):
            arr_mul_dft(a_2d, a_1d)
        with self.assertRaisesRegex(ValueError, "Both input arrays must be 1-dimensional."):
            arr_mul_dft(a_1d, a_2d)

    def test_raises_value_error_for_unknown_method(self):
        """Test that func_mul_dft raises ValueError for an unknown method."""
        with self.assertRaisesRegex(ValueError, "Unknown method: invalid_method"):
            arr_mul_dft(self.arr1_same, self.arr2_same, method="invalid_method")  # type: ignore

    def test_methods_are_equivalent_for_same_size_arrays(self):
        """Test that all methods produce similar results for same-sized arrays."""
        direct_result = arr_mul_dft(self.arr1_same, self.arr2_same, method="direct")
        for method in ["convolve", "convolve-fft"]:
            with self.subTest(method=method):
                conv_result = arr_mul_dft(self.arr1_same, self.arr2_same, method=method)  # type: ignore
                np.testing.assert_allclose(direct_result, conv_result, atol=1e-9)

    def test_methods_are_equivalent_for_different_size_arrays_no_antialias(self):
        """Test methods for different-sized arrays without antialiasing."""
        direct_result = arr_mul_dft(self.arr1_diff, self.arr2_diff, method="direct", antialias=False)
        for method in ["convolve", "convolve-fft"]:
            with self.subTest(method=method):
                conv_result = arr_mul_dft(self.arr1_diff, self.arr2_diff, method=method, antialias=False)  # type: ignore
                np.testing.assert_allclose(direct_result, conv_result, atol=1e-9)

    def test_methods_are_equivalent_for_different_size_arrays_with_antialias(self):
        """Test methods for different-sized arrays with antialiasing."""
        direct_result = arr_mul_dft(self.arr1_diff, self.arr2_diff, method="direct", antialias=True)
        for method in ["convolve", "convolve-fft"]:
            with self.subTest(method=method):
                conv_result = arr_mul_dft(self.arr1_diff, self.arr2_diff, method=method, antialias=True)  # type: ignore
                np.testing.assert_allclose(direct_result, conv_result, atol=1e-9)

    def test_direct_method_logic(self):
        """Test the logic of the 'direct' method with different scenarios."""
        # Same size
        expected_same = np.fft.fft(self.arr1_same * self.arr2_same)
        result_same = arr_mul_dft(self.arr1_same, self.arr2_same, method="direct")
        np.testing.assert_allclose(result_same, expected_same)

        # Different size, no antialias
        gx_padded = np.pad(self.arr2_diff, (0, self.arr1_diff.size - self.arr2_diff.size))
        expected_no_alias = np.fft.fft(self.arr1_diff * gx_padded)
        result_no_alias = arr_mul_dft(self.arr1_diff, self.arr2_diff, method="direct", antialias=False)
        np.testing.assert_allclose(result_no_alias, expected_no_alias)

        # Different size, with antialias
        n = self.arr1_diff.size + self.arr2_diff.size - 1
        fx_padded_alias = np.pad(self.arr1_diff, (0, n - self.arr1_diff.size))
        gx_padded_alias = np.pad(self.arr2_diff, (0, n - self.arr2_diff.size))
        expected_alias = np.fft.fft(fx_padded_alias * gx_padded_alias)
        result_alias = arr_mul_dft(self.arr1_diff, self.arr2_diff, method="direct", antialias=True)
        np.testing.assert_allclose(result_alias, expected_alias)

    def test_swapping_of_unequal_arrays(self):
        """Test that smaller array is correctly identified and padded."""
        # The function should swap fx and gx if gx is larger.
        # The result should be the same regardless of input order.
        res1 = arr_mul_dft(self.arr1_diff, self.arr2_diff, antialias=False)
        res2 = arr_mul_dft(self.arr2_diff, self.arr1_diff, antialias=False)
        np.testing.assert_allclose(res1, res2)

        res1_alias = arr_mul_dft(self.arr1_diff, self.arr2_diff, antialias=True)
        res2_alias = arr_mul_dft(self.arr2_diff, self.arr1_diff, antialias=True)
        np.testing.assert_allclose(res1_alias, res2_alias)
