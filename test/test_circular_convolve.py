import unittest

import numpy as np

from app.math import circular_convolve, gen_self_n_circ_convolve


class TestCircularConvolve(unittest.TestCase):
    """Test suite for the circular_convolve function using unittest."""

    def test_raises_value_error_for_non_1d_arrays(self):
        """Test that circular_convolve raises ValueError for non-1D inputs."""
        a_1d = np.array([1, 2, 3])
        a_2d = np.array([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "Both input arrays must be 1-dimensional."):
            circular_convolve(a_2d, a_1d)
        with self.assertRaisesRegex(ValueError, "Both input arrays must be 1-dimensional."):
            circular_convolve(a_1d, a_2d)

    def test_different_sized_arrays_fall_back_to_full_convolve(self):
        """Test fallback to np.convolve for arrays of different sizes."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2])
        expected = np.convolve(a, b, mode="full")
        result = circular_convolve(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_same_sized_arrays(self):
        """Test circular convolution for same-sized arrays with and without FFT."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        # Manual calculation:
        # np.convolve(a, b, 'full') -> [ 4, 13, 28, 27, 18]
        # res = [4, 13, 28]
        # res[:-1] += [27, 18] -> res becomes [4+27, 13+18, 28] -> [31, 31, 28]
        expected = np.array([31, 31, 28])
        for use_fft in [True, False]:
            with self.subTest(use_fft=use_fft):
                result = circular_convolve(a, b, use_fft=use_fft)
                np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_convolution_with_identity(self):
        """Test convolution with an identity element returns the original array."""
        a = np.array([1.5, 2.5, 3.5, 4.5])
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        for use_fft in [True, False]:
            with self.subTest(use_fft=use_fft):
                result = circular_convolve(a, identity, use_fft=use_fft)
                np.testing.assert_allclose(result, a, atol=1e-9)

    def test_convolution_with_zeros(self):
        """Test convolution with a zero array returns a zero array."""
        a = np.array([1, 2, 3, 4])
        b = np.zeros_like(a)
        expected = np.zeros_like(a)
        for use_fft in [True, False]:
            with self.subTest(use_fft=use_fft):
                result = circular_convolve(a, b, use_fft=use_fft)
                np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_fft_and_manual_methods_are_close(self):
        """Test that FFT and manual methods produce numerically close results."""
        np.random.seed(0)  # for reproducibility
        a = np.random.rand(100)
        b = np.random.rand(100)
        result_manual = circular_convolve(a, b, use_fft=False)
        result_fft = circular_convolve(a, b, use_fft=True)
        np.testing.assert_allclose(result_manual, result_fft, atol=1e-9)


class TestGenSelfNCircConvolve(unittest.TestCase):
    """Test suite for the gen_self_n_circ_convolve function."""

    def test_raises_value_error_for_non_1d_array(self):
        """Test that the generator raises ValueError for non-1D input."""
        arr_2d = np.array([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "Input array must be 1-dimensional"):
            # We need to consume the generator to trigger the exception
            next(gen_self_n_circ_convolve(arr_2d))

    def test_generator_yields_correct_convolutions(self):
        """Test that the generator yields correct successive convolutions."""
        arr = np.array([1, 2, 3])
        for use_fft in [True, False]:
            with self.subTest(use_fft=use_fft):
                gen = gen_self_n_circ_convolve(arr, use_fft=use_fft)

                # 1st yield: arr
                c1 = next(gen)
                np.testing.assert_allclose(c1, arr)

                # 2nd yield: circular_convolve(arr, arr)
                c2 = next(gen)
                expected_c2 = circular_convolve(arr, arr, use_fft=use_fft)
                np.testing.assert_allclose(c2, expected_c2)

                # 3rd yield: circular_convolve(c2, arr)
                c3 = next(gen)
                expected_c3 = circular_convolve(expected_c2, arr, use_fft=use_fft)
                np.testing.assert_allclose(c3, expected_c3)
