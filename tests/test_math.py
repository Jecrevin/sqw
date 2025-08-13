import unittest

import numpy as np

from sqw.math import continuous_fourier_transform


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


if __name__ == "__main__":
    unittest.main()
