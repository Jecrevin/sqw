import unittest

import numpy as np

from app.math import func_mul_ft, gen_func_self_n_mul_ft


class TestFuncMulFT(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, if any."""
        self.n = 128
        self.dx = 0.1
        self.x = np.arange(self.n) * self.dx
        self.fx = np.exp(-(self.x**2))
        self.gx = np.cos(self.x)

        self.fx_short = self.fx[: self.n // 2]
        self.gx_short = self.gx[: self.n // 2]

    def test_direct_method_same_size(self):
        """Test 'direct' method with same-sized arrays."""
        expected = np.fft.fft(self.fx * self.gx) * self.dx
        result = func_mul_ft(self.fx, self.gx, self.dx, method="direct")
        np.testing.assert_allclose(result, expected)

    def test_convolve_methods_same_size(self):
        """Test 'convolve' and 'convolve-fft' methods with same-sized arrays."""
        # According to convolution theorem: FT(f*g) = FT(f) * FT(g)
        # where * is convolution.
        # Here we compute FT(f*g) where * is element-wise product.
        # FT(f(x)g(x)) = 1/(2*pi) * (F(k) * G(k)) where F,G are FTs and * is convolution.
        # DFT(a*b) = 1/N * conv(DFT(a), DFT(b))
        # So func_mul_ft should give FT(f*g) = dx * DFT(fx*gx)
        # = dx * (1/N) * conv(DFT(fx), DFT(gx))
        # = dx * (1/N) * conv(Fw, Gw)
        # where Fw = DFT(fx), Gw = DFT(gx)
        direct_result = func_mul_ft(self.fx, self.gx, self.dx, method="direct")

        # Test 'convolve'
        convolve_result = func_mul_ft(self.fx, self.gx, self.dx, method="convolve")
        np.testing.assert_allclose(convolve_result, direct_result, atol=1e-14)

        # Test 'convolve-fft'
        convolve_fft_result = func_mul_ft(self.fx, self.gx, self.dx, method="convolve-fft")
        np.testing.assert_allclose(convolve_fft_result, direct_result, atol=1e-14)

    def test_direct_method_different_size(self):
        """Test 'direct' method with different-sized arrays."""
        # Test with antialias=False
        n_long = self.fx.size
        fx_padded = np.pad(self.fx_short, (0, n_long - self.fx_short.size))
        expected_no_alias = np.fft.fft(self.fx * fx_padded) * self.dx
        result_no_alias = func_mul_ft(self.fx, self.fx_short, self.dx, method="direct", antialias=False)
        np.testing.assert_allclose(result_no_alias, expected_no_alias)

        # Test with antialias=True
        n_padded = self.fx.size + self.fx_short.size - 1
        fx_padded_alias = np.pad(self.fx, (0, n_padded - self.fx.size))
        fx_short_padded_alias = np.pad(self.fx_short, (0, n_padded - self.fx_short.size))
        expected_alias = np.fft.fft(fx_padded_alias * fx_short_padded_alias) * self.dx
        result_alias = func_mul_ft(self.fx, self.fx_short, self.dx, method="direct", antialias=True)
        np.testing.assert_allclose(result_alias, expected_alias)

    def test_convolve_methods_different_size(self):
        """Test convolve methods with different-sized arrays."""
        direct_result_alias = func_mul_ft(self.fx, self.fx_short, self.dx, method="direct", antialias=True)

        convolve_result = func_mul_ft(self.fx, self.fx_short, self.dx, method="convolve", antialias=True)
        np.testing.assert_allclose(convolve_result, direct_result_alias, atol=1e-14)

        convolve_fft_result = func_mul_ft(self.fx, self.fx_short, self.dx, method="convolve-fft", antialias=True)
        np.testing.assert_allclose(convolve_fft_result, direct_result_alias, atol=1e-14)

    def test_analytical_square_gaussian(self):
        """
        Test FT of a squared Gaussian against its analytical formula.
        FT(exp(-x^2)) = sqrt(pi) * exp(-omega^2/4)
        """
        n = 256
        dx = 0.2
        x = (np.arange(n) - n / 2) * dx
        fx = np.exp(-(x**2) / 2)

        # The function whose FT is being calculated is f(x)^2 = exp(-x^2)
        # The analytical FT is sqrt(pi) * exp(-omega^2 / 4)
        omega = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        analytical_result = np.sqrt(np.pi) * np.exp(-(omega**2) / 4)

        for method in ["direct", "convolve", "convolve-fft"]:
            with self.subTest(method=method):
                numerical_result = func_mul_ft(fx, fx, dx, method=method)  # type: ignore
                # We shift both because omega from fftfreq is not monotonic
                np.testing.assert_allclose(
                    np.abs(np.fft.fftshift(numerical_result)),
                    np.fft.fftshift(analytical_result),
                    atol=1e-9,
                )

    def test_phase_consistency(self):
        """Test that the phase is consistent across different methods."""
        # Use 'direct' method as the reference
        direct_result = func_mul_ft(self.fx, self.gx, self.dx, method="direct")

        for method in ["convolve", "convolve-fft"]:
            with self.subTest(method=method):
                other_result = func_mul_ft(self.fx, self.gx, self.dx, method=method)  # type: ignore

                # 1. Compare complex values directly (checks magnitude and phase)
                np.testing.assert_allclose(other_result, direct_result, atol=1e-14)

                # 2. Explicitly compare phases where magnitude is significant
                direct_phase = np.angle(direct_result)
                other_phase = np.angle(other_result)
                magnitude = np.abs(direct_result)
                significant_indices = magnitude > 1e-9
                np.testing.assert_allclose(
                    direct_phase[significant_indices],
                    other_phase[significant_indices],
                    atol=1e-9,
                )


class TestGenFuncSelfNMulFT(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.n = 128
        self.dx = 0.1
        self.x = np.arange(self.n) * self.dx
        self.fx = np.exp(-(self.x**2) / 2)

    def test_direct_method(self):
        """Test the generator with the 'direct' method."""
        gen = gen_func_self_n_mul_ft(self.fx, self.dx, method="direct")

        # n=1: FT(f(x))
        expected1 = np.fft.fft(self.fx) * self.dx
        np.testing.assert_allclose(next(gen), expected1)

        # n=2: FT(f(x)^2)
        expected2 = np.fft.fft(self.fx**2) * self.dx
        np.testing.assert_allclose(next(gen), expected2)

        # n=3: FT(f(x)^3)
        expected3 = np.fft.fft(self.fx**3) * self.dx
        np.testing.assert_allclose(next(gen), expected3)

    def test_convolve_methods_consistency(self):
        """Test that convolve methods are consistent with the direct method."""
        num_terms = 3
        gen_direct = gen_func_self_n_mul_ft(self.fx, self.dx, method="direct")
        results_direct = [next(gen_direct) for _ in range(num_terms)]

        for method in ["convolve", "convolve-fft"]:
            with self.subTest(method=method):
                gen_other = gen_func_self_n_mul_ft(self.fx, self.dx, method=method)  # type: ignore
                results_other = [next(gen_other) for _ in range(num_terms)]
                np.testing.assert_allclose(results_other, results_direct, atol=1e-14)

    def test_analytical_square_gaussian(self):
        """
        Test FT of a squared Gaussian from the generator against its analytical formula.
        This corresponds to the second term from the generator (n=2).
        FT(exp(-x^2)) = sqrt(pi) * exp(-omega^2/4)
        """
        n = 256
        dx = 0.2
        x = (np.arange(n) - n / 2) * dx
        fx = np.exp(-(x**2) / 2)  # f(x) = exp(-x^2/2)

        # The function is f(x)^2 = exp(-x^2)
        # The analytical FT is sqrt(pi) * exp(-omega^2 / 4)
        omega = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        analytical_result = np.sqrt(np.pi) * np.exp(-(omega**2) / 4)

        for method in ["direct", "convolve", "convolve-fft"]:
            with self.subTest(method=method):
                gen = gen_func_self_n_mul_ft(fx, dx, method=method)  # type: ignore
                next(gen)  # Skip n=1
                numerical_result = next(gen)  # n=2: FT(f(x)^2)

                # We shift both because omega from fftfreq is not monotonic
                np.testing.assert_allclose(
                    np.abs(np.fft.fftshift(numerical_result)),
                    np.fft.fftshift(analytical_result),
                    atol=1e-9,
                )

    def test_phase_consistency(self):
        """Test that the phase is consistent across different methods for the generator."""
        num_terms = 3
        gen_direct = gen_func_self_n_mul_ft(self.fx, self.dx, method="direct")
        results_direct = [next(gen_direct) for _ in range(num_terms)]

        for method in ["convolve", "convolve-fft"]:
            with self.subTest(method=method):
                gen_other = gen_func_self_n_mul_ft(self.fx, self.dx, method=method)  # type: ignore
                results_other = [next(gen_other) for _ in range(num_terms)]

                for i in range(num_terms):
                    direct_result = results_direct[i]
                    other_result = results_other[i]

                    # Explicitly compare phases where magnitude is significant
                    direct_phase = np.angle(direct_result)
                    other_phase = np.angle(other_result)
                    magnitude = np.abs(direct_result)
                    significant_indices = magnitude > 1e-9
                    np.testing.assert_allclose(
                        direct_phase[significant_indices],
                        other_phase[significant_indices],
                        atol=1e-9,
                        err_msg=f"Phase mismatch for term {i + 1}",
                    )
