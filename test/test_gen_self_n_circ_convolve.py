import unittest
from collections.abc import Generator
import numpy as np
from app.math import gen_self_n_circ_convolve


class TestGenSelfNCircConvolve(unittest.TestCase):
    def test_non_1d_array_raises_error(self):
        """Test that a non-1D array raises a ValueError."""
        with self.assertRaisesRegex(ValueError, "Input array must be one-dimensional."):
            # The generator is lazy, so we need to try to get a value from it.
            next(gen_self_n_circ_convolve(np.array([[1, 2], [3, 4]])))

    def test_returns_generator(self):
        """Test that the function returns a generator object."""
        fx = np.array([1, 0, 0])
        result = gen_self_n_circ_convolve(fx)
        self.assertIsInstance(result, Generator)

    def test_delta_function_input(self):
        """Test convolution with a delta function at the origin."""
        fx = np.array([1.0, 0.0, 0.0, 0.0])
        gen = gen_self_n_circ_convolve(fx)

        # The n-th self-convolution of a delta function is the delta function itself.
        for _ in range(5):
            result = next(gen)
            np.testing.assert_allclose(result, fx, atol=1e-9)

    def test_shifted_delta_function_input(self):
        """Test convolution with a shifted delta function."""
        fx = np.array([0.0, 1.0, 0.0, 0.0])
        gen = gen_self_n_circ_convolve(fx)

        # 1st convolution (fx * fx)
        expected1 = np.array([0.0, 0.0, 1.0, 0.0])
        result1 = next(gen)
        np.testing.assert_allclose(result1, expected1, atol=1e-9)

        # 2nd convolution (fx * fx * fx)
        expected2 = np.array([0.0, 0.0, 0.0, 1.0])
        result2 = next(gen)
        np.testing.assert_allclose(result2, expected2, atol=1e-9)

        # 3rd convolution (fx * fx * fx * fx)
        expected3 = np.array([1.0, 0.0, 0.0, 0.0])
        result3 = next(gen)
        np.testing.assert_allclose(result3, expected3, atol=1e-9)

        # 4th convolution (fx * fx * fx * fx * fx)
        expected4 = np.array([0.0, 1.0, 0.0, 0.0])  # Back to the original fx
        result4 = next(gen)
        np.testing.assert_allclose(result4, expected4, atol=1e-9)

    def test_constant_function_input(self):
        """Test convolution with a constant function."""
        fx = np.array([2.0, 2.0, 2.0, 2.0])
        gen = gen_self_n_circ_convolve(fx)

        # The convolution of a constant function with itself is proportional to the constant function.
        # C(x) = c, (C * C)(x) = sum(C(y) * C(x-y)) = sum(c * c) = N * c^2
        # Here N=4, c=2. So N*c^2 = 4 * 4 = 16.
        # The result of the first convolution should be a constant array of 16.
        expected1 = np.array([16.0, 16.0, 16.0, 16.0])
        result1 = next(gen)
        np.testing.assert_allclose(result1, expected1, atol=1e-9)

        # The next convolution is with the original fx.
        # (C*C) * C -> res = [16,16,16,16], fx = [2,2,2,2]
        # sum(res(y) * fx(x-y)) = sum(16 * 2) = N * 16 * 2 = 4 * 32 = 128
        expected2 = np.array([128.0, 128.0, 128.0, 128.0])
        result2 = next(gen)
        np.testing.assert_allclose(result2, expected2, atol=1e-9)

    def test_general_input(self):
        """Test convolution with a general array."""
        fx = np.array([1.0, 2.0, 3.0])
        gen = gen_self_n_circ_convolve(fx)

        # Manual calculation of circular convolution:
        # c[k] = sum_{j=0}^{N-1} a[j] * b[(k-j) mod N]
        # fx * fx:
        # c[0] = 1*1 + 2*3 + 3*2 = 1 + 6 + 6 = 13
        # c[1] = 1*2 + 2*1 + 3*3 = 2 + 2 + 9 = 13
        # c[2] = 1*3 + 2*2 + 3*1 = 3 + 4 + 3 = 10
        expected1 = np.array([13.0, 13.0, 10.0])
        result1 = next(gen)
        np.testing.assert_allclose(result1, expected1, atol=1e-9)

        # res * fx: res=[13, 13, 10], fx=[1, 2, 3]
        # c[0] = 13*1 + 13*3 + 10*2 = 13 + 39 + 20 = 72
        # c[1] = 13*2 + 13*1 + 10*3 = 26 + 13 + 30 = 69
        # c[2] = 13*3 + 13*2 + 10*1 = 39 + 26 + 10 = 75
        expected2 = np.array([72.0, 69.0, 75.0])
        result2 = next(gen)
        np.testing.assert_allclose(result2, expected2, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
