"""Test cases for math utilities in sqw._math module."""

import unittest

import numpy as np

from sqw._math import (
    continuous_fourier_transform,
    is_all_array_1d,
    is_all_array_linspace,
    is_array_linspace,
    linear_convolve,
    linear_convolve_x_axis,
    self_linear_convolve,
    self_linear_convolve_x_axis,
    trim_function,
)


class TestMath(unittest.TestCase):
    """Test cases for math utilities in sqw._math module."""

    def test_is_all_array_1d(self):
        """Test `is_all_array_1d` function."""
        self.assertTrue(is_all_array_1d(np.array([1, 2]), np.array([3, 4])))
        self.assertFalse(is_all_array_1d(np.array([1, 2]), np.array([[3, 4]])))
        self.assertTrue(is_all_array_1d())

    def test_is_array_linspace(self):
        """Test `is_array_linspace` function."""
        self.assertTrue(is_array_linspace(np.array([1, 2, 3])))
        self.assertTrue(is_array_linspace(np.array([1.0, 1.5, 2.0])))
        self.assertFalse(is_array_linspace(np.array([1, 2, 4])))
        self.assertFalse(is_array_linspace(np.array([1.0, 1.5, 2.1])))
        self.assertTrue(is_array_linspace(np.array([1])))
        self.assertTrue(is_array_linspace(np.array([1, 2])))
        # Test with extreme small step
        self.assertTrue(is_array_linspace(np.array([0, 1e-16, 2e-16])))
        self.assertFalse(is_array_linspace(np.array([0, 1e-16, 2.0001e-16])))
        # Test with extreme large step
        self.assertTrue(is_array_linspace(np.array([0, 1e16, 2e16])))
        self.assertFalse(is_array_linspace(np.array([0, 1e16, 2.0001e16])))

    def test_is_all_array_linspace(self):
        """Test `is_all_array_linspace` function."""
        self.assertTrue(is_all_array_linspace(np.array([1, 2, 3]), np.linspace(0, 1, 5)))
        self.assertFalse(is_all_array_linspace(np.array([1, 2, 3]), np.array([1, 3, 4])))
        self.assertTrue(is_all_array_linspace())

    def test_linear_convolve(self):
        """Test `linear_convolve` function."""
        # Test convolution of two box functions, which results in a triangle function
        dx = 0.01
        x = np.arange(-1, 1 + dx, dx)
        fx = np.zeros_like(x)
        fx[np.abs(x) <= 0.5] = 1.0

        result = linear_convolve(fx, fx, dx)

        # Analytical result
        x_conv = self_linear_convolve_x_axis(x)
        analytical = np.maximum(0, 1 - np.abs(x_conv))

        np.testing.assert_allclose(result, analytical, atol=dx)

    def test_self_linear_convolve(self):
        """Test `self_linear_convolve` function."""
        fx = np.array([1.0, 2.0, 1.0])
        dx = 0.5
        expected = linear_convolve(fx, fx, dx)
        result = self_linear_convolve(fx, dx)
        np.testing.assert_allclose(result, expected)

    def test_linear_convolve_x_axis(self):
        """Test `linear_convolve_x_axis` function."""
        x1 = np.linspace(0, 1, 11)
        x2 = np.linspace(0.5, 1.5, 21)
        result = linear_convolve_x_axis(x1, x2)
        expected = np.linspace(0.5, 2.5, 11 + 21 - 1)
        np.testing.assert_allclose(result, expected)

    def test_self_linear_convolve_x_axis(self):
        """Test `self_linear_convolve_x_axis` function."""
        x = np.linspace(-1, 1, 101)
        result = self_linear_convolve_x_axis(x)
        expected = linear_convolve_x_axis(x, x)
        np.testing.assert_allclose(result, expected)

    def test_continuous_fourier_transform(self):
        """Test `continuous_fourier_transform` function."""
        # Test with a Gaussian function, whose Fourier transform is also a Gaussian.
        # f(t) = exp(-a * t^2) -> F(w) = sqrt(pi/a) * exp(-w^2 / (4a))
        a = 0.5
        dt = 0.01
        time = np.arange(-20, 20, dt)

        # Test with real signal
        signal_real = np.exp(-a * time**2)
        freq, ft_signal = continuous_fourier_transform(time, signal_real)
        analytical_ft = np.sqrt(np.pi / a) * np.exp(-(freq**2) / (4 * a))
        print(np.max(analytical_ft))
        np.testing.assert_allclose(ft_signal, analytical_ft, atol=1e-9)

        # Test with complex signal (shifted Gaussian)
        # f(t) = exp(-a * t^2) * exp(i*w0*t) -> F(w) = sqrt(pi/a) * exp(-(w-w0)^2 / (4a))
        w0 = 5.0
        signal_complex = signal_real * np.exp(1j * w0 * time)
        freq_c, ft_signal_c = continuous_fourier_transform(time, signal_complex)
        analytical_ft_c = np.sqrt(np.pi / a) * np.exp(-((freq_c - w0) ** 2) / (4 * a))
        np.testing.assert_allclose(ft_signal_c, analytical_ft_c, atol=1e-9)

    def test_trim_function(self):
        """Test `trim_function` function."""
        x = np.linspace(-10, 10, 101)
        y = np.zeros(101)
        y[40:61] = np.sin(np.linspace(0, np.pi, 21))  # Peak at index 50

        cut_ratio = 0.01

        x_trimmed, y_trimmed = trim_function(x, y, cut_ratio)

        # Max value is 1.0. Threshold is 0.01.
        # sin(x) > 0.01 for x in approx (0.01, pi-0.01)
        # Indices should be around 40 and 60
        significant_indices = np.nonzero(y > 1.0 * cut_ratio)[0]
        start, end = significant_indices[0], significant_indices[-1] + 1

        np.testing.assert_array_equal(x_trimmed, x[start:end])
        np.testing.assert_array_equal(y_trimmed, y[start:end])

        # Test with complex values
        y_complex = y * (1 + 1j)
        x_trimmed_c, y_trimmed_c = trim_function(x, y_complex, cut_ratio)
        # abs(y_complex) = y * sqrt(2). max is sqrt(2). threshold is sqrt(2)*0.01
        # The indices should be the same
        np.testing.assert_array_equal(x_trimmed_c, x[start:end])
        np.testing.assert_array_equal(y_trimmed_c, y_complex[start:end])


if __name__ == "__main__":
    unittest.main()
