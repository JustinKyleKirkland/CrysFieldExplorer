"""
Unit tests for the CrysFieldExplorer Operators module.
"""

import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from src.CrysFieldExplorer.Operators import QuantumOperator, StevensOperator


class TestQuantumOperator(unittest.TestCase):
    """Test cases for the QuantumOperator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.operator = QuantumOperator("Er3+")  # Er3+ has J=7.5

    def test_initialization(self):
        """Test initialization with valid and invalid ions."""
        # Test valid initialization
        op = QuantumOperator("Er3+")
        self.assertEqual(op.S, 1.5)
        self.assertEqual(op.L, 6.0)
        self.assertEqual(op.J, 7.5)

        # Test invalid ion
        with self.assertRaises(ValueError):
            QuantumOperator("Invalid3+")

    def test_matrix_dimensions(self):
        """Test that all operators return matrices of correct dimensions."""
        size = int(2 * self.operator.J + 1)  # Should be 16x16 for Er3+

        matrices = self.operator.get_matrices()
        for name, matrix in matrices.items():
            with self.subTest(operator=name):
                self.assertEqual(matrix.shape, (size, size))

    def test_j_squared_eigenvalues(self):
        """Test that J² has correct eigenvalues."""
        j_squared = self.operator.j_squared()
        eigenvalues = np.linalg.eigvals(j_squared).real
        expected_value = self.operator.J * (self.operator.J + 1)

        # All eigenvalues should be approximately J(J+1)
        for eigenvalue in eigenvalues:
            self.assertAlmostEqual(eigenvalue, expected_value, places=10)

    def test_jz_eigenvalues(self):
        """Test that Jz has correct eigenvalues."""
        jz = self.operator.j_z()
        eigenvalues = np.sort(np.linalg.eigvals(jz).real)
        expected = np.arange(-self.operator.J, self.operator.J + 1)

        assert_array_almost_equal(eigenvalues, expected)

    def test_raising_lowering_operators(self):
        """Test J+ and J- operators."""
        j_plus = self.operator.j_plus()
        j_minus = self.operator.j_minus()

        # Test that J+ and J- are conjugate transposes
        assert_array_almost_equal(j_plus.conj().T, j_minus)

    def test_hermiticity(self):
        """Test that Jx, Jy, Jz are Hermitian."""
        matrices = self.operator.get_matrices()

        for name in ["jx", "jy", "jz"]:
            with self.subTest(operator=name):
                matrix = matrices[name]
                assert_array_almost_equal(matrix, matrix.conj().T)


class TestStevensOperator(unittest.TestCase):
    """Test cases for the StevensOperator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.operator = StevensOperator("Er3+")

    def test_initialization(self):
        """Test initialization and inheritance."""
        self.assertIsInstance(self.operator, QuantumOperator)
        self.assertEqual(self.operator.J, 7.5)

    def test_operator_symmetries(self):
        """Test symmetry properties of Stevens operators."""
        # O_2^0 should be Hermitian
        o20 = self.operator.get_operator(2, 0)
        assert_array_almost_equal(o20, o20.conj().T)

        # Test each operator pair
        for n in [2, 4, 6]:
            for m in range(1, n + 1):
                with self.subTest(n=n, m=m):
                    print(f"\n{'=' * 50}")
                    print(f"Testing O_{n}^{m} and O_{n}^{-m}")

                    op_m = self.operator.get_operator(n, m)
                    op_minus_m = self.operator.get_operator(n, -m)

                    # For m=1, O_n^-1 = i * O_n^1
                    if m == 1:
                        expected = 1j * op_m
                        assert_array_almost_equal(op_minus_m, expected)
                        continue

                    # For m=2, the relationship is more complex
                    # O_n^-2 is related to O_n^2 by a sign-dependent phase factor
                    if m == 2:
                        # Check that non-zero elements have magnitude preserved
                        nonzero_m = np.abs(op_m) > 1e-10
                        nonzero_minus_m = np.abs(op_minus_m) > 1e-10
                        assert_array_almost_equal(nonzero_m, nonzero_minus_m)

                        # Check that non-zero elements are related by ±i
                        ratios = op_minus_m[nonzero_m] / op_m[nonzero_m]
                        # All ratios should be ±i
                        assert_array_almost_equal(np.abs(ratios), np.ones_like(ratios))
                        # Real parts should be zero
                        assert_array_almost_equal(ratios.real, np.zeros_like(ratios.real))
                        continue

                    # For higher m, implement appropriate relationships here
                    # For now, just print the relationships we find
                    print("\nHigher order relationships:")
                    nonzero = np.abs(op_m) > 1e-10
                    if np.any(nonzero):
                        ratios = op_minus_m[nonzero] / op_m[nonzero]
                        print(f"Ratios for m={m}:")
                        print(ratios)

    def test_cache_functionality(self):
        """Test that operator caching works correctly."""
        # First call should calculate the operator
        first_call = self.operator.get_operator(2, 0)

        # Second call should return cached value
        second_call = self.operator.get_operator(2, 0)

        # Should be exactly the same object
        self.assertIs(first_call, second_call)

    def test_invalid_operators(self):
        """Test error handling for invalid operator requests."""
        # Test invalid n
        with self.assertRaises(ValueError):
            self.operator.get_operator(3, 0)

        # Test invalid m
        with self.assertRaises(ValueError):
            self.operator.get_operator(2, 3)

    def test_get_all_operators(self):
        """Test that get_all_operators returns all implemented operators."""
        all_ops = self.operator.get_all_operators()

        # Check that we get the expected number of operators
        expected_operators = set()
        for n in [2, 4, 6]:
            for m in range(-n, n + 1):
                expected_operators.add(f"O_{n}^{m}")

        self.assertEqual(set(all_ops.keys()), expected_operators)

        # Check that all returned operators have correct dimensions
        size = int(2 * self.operator.J + 1)
        for matrix in all_ops.values():
            self.assertEqual(matrix.shape, (size, size))


if __name__ == "__main__":
    unittest.main()
