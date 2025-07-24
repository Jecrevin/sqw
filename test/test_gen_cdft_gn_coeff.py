import math
import unittest
from collections.abc import Generator
from itertools import islice
import numpy as np
from app.math import _gen_cdft_gn_coeff


class TestGenCdftGnCoeff(unittest.TestCase):
    """Tests for the _gen_cdft_gn_coeff generator function."""

    def test_returns_generator(self):
        """Test that the function returns a generator object."""
        self.assertIsInstance(_gen_cdft_gn_coeff(1.0), Generator)

    def test_positive_integer_val(self):
        """Test the generator with a positive integer value."""
        val = 2
        gen = _gen_cdft_gn_coeff(val)
        expected_sequence = [
            val**0 / math.factorial(0),  # 1.0
            val**1 / math.factorial(1),  # 2.0
            val**2 / math.factorial(2),  # 2.0
            val**3 / math.factorial(3),  # 1.333...
            val**4 / math.factorial(4),  # 0.666...
        ]
        generated_sequence = list(islice(gen, 5))

        for expected, generated in zip(expected_sequence, generated_sequence):
            self.assertAlmostEqual(expected, generated)

    def test_positive_float_val(self):
        """Test the generator with a positive float value."""
        val = 1.5
        gen = _gen_cdft_gn_coeff(val)
        expected_sequence = [
            val**0 / math.factorial(0),  # 1.0
            val**1 / math.factorial(1),  # 1.5
            val**2 / math.factorial(2),  # 1.125
            val**3 / math.factorial(3),  # 0.5625
        ]
        generated_sequence = list(islice(gen, 4))

        for expected, generated in zip(expected_sequence, generated_sequence):
            self.assertAlmostEqual(expected, generated)

    def test_zero_val(self):
        """Test the generator with val = 0."""
        gen = _gen_cdft_gn_coeff(0.0)
        # The sequence should be 1.0, 0.0, 0.0, 0.0, ...
        self.assertAlmostEqual(next(gen), 1.0)
        for _ in range(10):
            self.assertAlmostEqual(next(gen), 0.0)

    def test_one_val(self):
        """Test the generator with val = 1."""
        gen = _gen_cdft_gn_coeff(1.0)
        # The sequence should be 1/n!, which is 1, 1, 0.5, 0.166..., etc.
        expected_sequence = [1.0 / math.factorial(n) for n in range(5)]
        generated_sequence = list(islice(gen, 5))

        for expected, generated in zip(expected_sequence, generated_sequence):
            self.assertAlmostEqual(expected, generated)

    def test_negative_val(self):
        """Test the generator with a negative value."""
        val = -2.0
        gen = _gen_cdft_gn_coeff(val)
        # The sequence should be (-2)^n / n!, which alternates in sign.
        expected_sequence = [
            (-2.0) ** 0 / math.factorial(0),  # 1.0
            (-2.0) ** 1 / math.factorial(1),  # -2.0
            (-2.0) ** 2 / math.factorial(2),  # 2.0
            (-2.0) ** 3 / math.factorial(3),  # -1.333...
        ]
        generated_sequence = list(islice(gen, 4))

        for expected, generated in zip(expected_sequence, generated_sequence):
            self.assertAlmostEqual(expected, generated)

    def test_numpy_number_input(self):
        """Test the generator with a numpy.number type as input."""
        val = np.float64(3.0)
        gen = _gen_cdft_gn_coeff(val)
        expected_sequence = [
            val**0 / math.factorial(0),  # 1.0
            val**1 / math.factorial(1),  # 3.0
            val**2 / math.factorial(2),  # 4.5
        ]
        generated_sequence = list(islice(gen, 3))

        for expected, generated in zip(expected_sequence, generated_sequence):
            self.assertAlmostEqual(expected, generated)
            self.assertIsInstance(generated, float)


if __name__ == "__main__":
    unittest.main()
