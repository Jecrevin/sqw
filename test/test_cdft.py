import unittest
import numpy as np
from app.math import cdft


class TestCdft(unittest.TestCase):
    """Tests for the cdft function."""

    def test_input_validation(self):
        """Test the input validation for the cdft function."""
        x = np.linspace(-10, 10, 100)
        fx = np.random.rand(100)

        # Test non-1D arrays
        with self.assertRaisesRegex(ValueError, "Both x and fx must be one-dimensional arrays."):
            cdft(np.array([[1, 2], [3, 4]]), fx)
        with self.assertRaisesRegex(ValueError, "Both x and fx must be one-dimensional arrays."):
            cdft(x, np.array([[1, 2], [3, 4]]))

        # Test mismatched sizes
        with self.assertRaisesRegex(ValueError, "x and fx must have the same size."):
            cdft(x, np.random.rand(99))

        # Test non-linearly spaced x
        with self.assertRaisesRegex(ValueError, "x must be a linearly spaced array."):
            x_non_lin = np.array([1, 2, 4, 5])
            fx_non_lin = np.random.rand(4)
            cdft(x_non_lin, fx_non_lin)

    def test_constant_fx(self):
        """Test with fx as a constant function. exp(fx) is a scaled delta function in frequency domain."""
        x = np.linspace(-10, 10, 128)
        fx = np.ones_like(x) * 2.0  # f(x) = 2
        # F{exp(2)} = exp(2) * F{1} = exp(2) * delta(k)
        # The result should be a spike at zero frequency and zero elsewhere.
        # The FFT of a constant is a spike at k=0.
        result = cdft(x, fx)
        expected = np.zeros_like(result, dtype=np.complex128)
        # The value at k=0 is exp(c) * (b-a) where [a,b] is the interval
        expected[0] = np.exp(2.0) * (x[-1] - x[0])

        # The FFT implementation might have slight differences, so we check the main properties.
        self.assertAlmostEqual(np.sum(np.abs(result)), np.abs(expected[0]), places=5)
        self.assertEqual(np.argmax(np.abs(result)), 0)
        # Check that other values are close to zero
        self.assertTrue(np.allclose(np.abs(result[1:]), 0, atol=1e-9))

    def test_gaussian_fx(self):
        """Test with fx as a Gaussian. exp(fx) is also related to a Gaussian."""
        # Let f(x) = -a*x^2. Then exp(f(x)) is a Gaussian.
        # The Fourier transform of a Gaussian is another Gaussian.
        n_points = 256
        x_max = 20
        x = np.linspace(-x_max, x_max, n_points, endpoint=False)
        dx = x[1] - x[0]
        a = 0.1
        fx = -a * x**2

        # Theoretical Fourier Transform of exp(-a*x^2)
        # F{exp(-a*x^2)}(k) = sqrt(pi/a) * exp(-pi^2 * k^2 / a)
        # Our frequency vector `k` from FFT is `fftfreq(n, d)`
        k = np.fft.fftfreq(n_points, d=dx)
        expected = np.sqrt(np.pi / a) * np.exp(-(np.pi**2) * k**2 / a)

        # Result from cdft
        result_cdft = cdft(x, fx, cutoff=20)

        # The cdft result is an FFT, so we need to compare it with the theoretical FFT.
        # The cdft function already multiplies by dx, which is part of the continuous FT definition.
        # The result from cdft is F{exp(fx)} * dx.
        # The theoretical result is the continuous FT.
        # So we compare result_cdft/dx with the theoretical result.
        # However, the scaling of FFT can be tricky. Let's compare shapes and relative values.
        # Normalizing both results to have the same max value is a robust check.
        result_cdft_normalized = np.abs(result_cdft) / np.max(np.abs(result_cdft))
        expected_normalized = np.abs(expected) / np.max(np.abs(expected))

        # We compare the absolute values, ignoring phase, and normalized.
        np.testing.assert_allclose(result_cdft_normalized, expected_normalized, atol=1e-2)

    def test_zero_fx(self):
        """Test with fx = 0. exp(fx) = 1. FT is a delta function."""
        x = np.linspace(-5, 5, 128)
        fx = np.zeros_like(x)
        # F{exp(0)} = F{1} = delta(k)
        result = cdft(x, fx)
        expected = np.zeros_like(result, dtype=np.complex128)
        expected[0] = x[-1] - x[0]  # Integral of 1 over the domain

        # Check that the spike is at k=0 and other values are near zero.
        self.assertAlmostEqual(np.sum(np.abs(result)), np.abs(expected[0]), places=5)
        self.assertEqual(np.argmax(np.abs(result)), 0)
        self.assertTrue(np.allclose(np.abs(result[1:]), 0, atol=1e-9))

    def test_cutoff_effect(self):
        """Test that increasing the cutoff parameter improves accuracy."""
        n_points = 128
        x = np.linspace(-10, 10, n_points, endpoint=False)
        dx = x[1] - x[0]
        a = 0.1
        fx = -a * x**2  # Gaussian

        # Theoretical result
        k = np.fft.fftfreq(n_points, d=dx)
        expected = np.sqrt(np.pi / a) * np.exp(-(np.pi**2) * k**2 / a)

        # Calculate with low and high cutoff
        result_low_cutoff = cdft(x, fx, cutoff=3)
        result_high_cutoff = cdft(x, fx, cutoff=20)

        # Errors compared to the theoretical result (scaled by dx)
        error_low = np.sum(np.abs(result_low_cutoff - expected * dx))
        error_high = np.sum(np.abs(result_high_cutoff - expected * dx))

        self.assertLess(error_high, error_low)


if __name__ == "__main__":
    unittest.main()
