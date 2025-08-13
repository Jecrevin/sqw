import unittest

import numpy as np

from sqw.math import (
    continuous_fourier_transform,
    even_extend,
    is_linspace,
    linear_convolve,
    linear_convolve_x_axis,
    odd_extend,
    self_linear_convolve,
    self_linear_convolve_x_axis,
)


class TestLinearConvolve(unittest.TestCase):
    def test_linear_convolve(self):
        fx = np.array([1, 2, 3])
        gx = np.array([1, 1])
        dx = 0.1
        expected = np.convolve(fx, gx, mode="full") * dx
        result = linear_convolve(fx, gx, dx)
        np.testing.assert_allclose(result, expected)

        # Test with np.floating
        dx_np = np.float64(0.1)
        result_np = linear_convolve(fx, gx, dx_np)
        np.testing.assert_allclose(result_np, expected)

    def test_self_linear_convolve(self):
        fx = np.array([1, 2, 3])
        dx = 0.1
        expected = np.convolve(fx, fx, mode="full") * dx
        result = self_linear_convolve(fx, dx)
        np.testing.assert_allclose(result, expected)

        # Test with np.floating
        dx_np = np.float64(0.1)
        result_np = self_linear_convolve(fx, dx_np)
        np.testing.assert_allclose(result_np, expected)


class TestLinearConvolveXAxis(unittest.TestCase):
    def test_linear_convolve_x_axis(self):
        x1 = np.linspace(0, 1, 11)
        x2 = np.linspace(0.5, 1.5, 11)
        result = linear_convolve_x_axis(x1, x2)
        expected = np.linspace(0.5, 2.5, 21)
        np.testing.assert_allclose(result, expected)

        # Test exceptions
        with self.assertRaisesRegex(ValueError, "Input arrays must be one-dimensional."):
            linear_convolve_x_axis(np.array([[1]]), x2)
        with self.assertRaisesRegex(ValueError, "Input arrays must be one-dimensional."):
            linear_convolve_x_axis(x1, np.array([[1]]))

        with self.assertRaisesRegex(ValueError, "Input arrays must be linear spaced."):
            linear_convolve_x_axis(np.array([1, 2, 4]), x2)
        with self.assertRaisesRegex(ValueError, "Input arrays must be linear spaced."):
            linear_convolve_x_axis(x1, np.array([1, 2, 4]))

        with self.assertRaisesRegex(ValueError, "Input arrays must have the same step size."):
            linear_convolve_x_axis(x1, np.linspace(0.5, 1.5, 12))

    def test_self_linear_convolve_x_axis(self):
        x = np.linspace(0, 1, 11)
        result = self_linear_convolve_x_axis(x)
        expected = np.linspace(0, 2, 21)
        np.testing.assert_allclose(result, expected)


class TestIsLinspace(unittest.TestCase):
    def test_is_linspace(self):
        self.assertTrue(is_linspace(np.linspace(0, 10, 100)))
        self.assertTrue(is_linspace(np.array([1, 2, 3])))
        self.assertTrue(is_linspace(np.array([1, 5])))
        self.assertTrue(is_linspace(np.array([1])))
        self.assertTrue(is_linspace(np.array([])))
        self.assertFalse(is_linspace(np.array([1, 2, 4])))
        self.assertFalse(is_linspace(np.array([1.0, 2.0, 3.1])))
        with self.assertRaisesRegex(ValueError, "Input array must be one-dimensional."):
            is_linspace(np.array([[1, 2], [3, 4]]))


class TestExtendFunctions(unittest.TestCase):
    def test_odd_extend(self):
        arr = np.array([0, 1, 2, 3])
        expected = np.array([-3, -2, -1, 0, 1, 2, 3])
        result = odd_extend(arr)
        np.testing.assert_array_equal(result, expected)

        # Test with non-zero start
        arr_non_zero = np.array([1, 2, 3])
        expected_non_zero = np.array([-3, -2, 1, 2, 3])
        result_non_zero = odd_extend(arr_non_zero)
        np.testing.assert_array_equal(result_non_zero, expected_non_zero)

    def test_even_extend(self):
        arr = np.array([0, 1, 2, 3])
        expected = np.array([3, 2, 1, 0, 1, 2, 3])
        result = even_extend(arr)
        np.testing.assert_array_equal(result, expected)

        # Test with non-zero start
        arr_non_zero = np.array([1, 2, 3])
        expected_non_zero = np.array([3, 2, 1, 2, 3])
        result_non_zero = even_extend(arr_non_zero)
        np.testing.assert_array_equal(result_non_zero, expected_non_zero)


class TestContinuousFourierTransform(unittest.TestCase):
    def test_gaussian_fourier_transform(self):
        # Parameters
        N = 512
        L = 20.0
        sigma = 1.0
        x = np.linspace(-L / 2, L / 2, N)
        dx = x[1] - x[0]
        sampling_rate = 1.0 / dx

        # Gaussian signal
        signal = np.exp(-(x**2) / (2 * sigma**2))

        # Analytical FT of Gaussian: sqrt(2*pi)*sigma*exp(-2*pi^2*sigma^2*f^2)
        f = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        analytical = np.sqrt(2 * np.pi) * sigma * np.exp(-2 * (np.pi**2) * (sigma**2) * (f**2))

        # Numerical FT
        numerical = continuous_fourier_transform(signal, sampling_rate)
        numerical_magnitude = np.abs(numerical)

        # Normalize for comparison
        analytical /= analytical.max()
        numerical_magnitude /= numerical_magnitude.max()

        # Compare (allowing for numerical error)
        np.testing.assert_allclose(numerical_magnitude, analytical, rtol=1e-2, atol=1e-2)

    def test_exceptions(self):
        with self.assertRaisesRegex(ValueError, "Input signal must be one-dimensional."):
            continuous_fourier_transform(np.array([[1, 2], [3, 4]]), 1.0)


if __name__ == "__main__":
    unittest.main()
