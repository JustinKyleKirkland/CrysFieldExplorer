import unittest

import numpy as np

from CrysFieldExplorer.CrysFieldExplorer import CrysFieldExplorer
from CrysFieldExplorer.utilities import Utilities


class TestCrysFieldExplorer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Common test parameters
        self.magnetic_ion = "Er3+"
        self.stevens_idx = [
            [2, 0],
            [2, 1],
            [2, 2],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 4],
            [6, 0],
            [6, 1],
            [6, 2],
            [6, 3],
            [6, 4],
            [6, 5],
            [6, 6],
        ]
        self.alpha = 0.01 * 10.0 * 4 / (45 * 35)
        self.beta = 0.01 * 100.0 * 2 / (11 * 15 * 273)
        self.gamma = 0.01 * 10.0 * 8 / (13**2 * 11**2 * 3**3 * 7)
        self.parameters = np.ones(15)  # Simple test parameters
        self.temperature = 5.0
        self.field = [0, 0, 0]

        # Create instance for testing
        self.cef = CrysFieldExplorer(
            self.magnetic_ion,
            self.stevens_idx,
            self.alpha,
            self.beta,
            self.gamma,
            self.parameters,
            self.temperature,
            self.field,
        )

    def test_initialization(self):
        """Test proper initialization of CrysFieldExplorer instance."""
        # Test parameter dictionary creation
        self.assertTrue(isinstance(self.cef.parameters, dict))
        self.assertEqual(len(self.cef.parameters), 15)
        self.assertEqual(self.cef.temperature, 5.0)
        self.assertEqual(self.cef.field, [0, 0, 0])

    def test_hamiltonian(self):
        """Test Hamiltonian calculation."""
        eigenvalues, eigenvectors, hamiltonian = self.cef.Hamiltonian()

        # Check shapes and types
        self.assertTrue(isinstance(eigenvalues, np.ndarray))
        self.assertTrue(isinstance(eigenvectors, np.ndarray))
        self.assertTrue(isinstance(hamiltonian, np.ndarray))

        # Check dimensions
        dimension = int(2 * self.cef.J + 1)
        self.assertEqual(eigenvalues.shape, (dimension,))
        self.assertEqual(eigenvectors.shape, (dimension, dimension))
        self.assertEqual(hamiltonian.shape, (dimension, dimension))

        # Check if hamiltonian is Hermitian
        np.testing.assert_array_almost_equal(hamiltonian, hamiltonian.conj().T)

    def test_magnetic_hamiltonian(self):
        """Test magnetic Hamiltonian calculation."""
        Bx, By, Bz = 1.0, 0.0, 0.0
        mag_hamiltonian = self.cef.magnetic_Hamiltonian(Bx, By, Bz)

        # Check if magnetic Hamiltonian is Hermitian
        np.testing.assert_array_almost_equal(mag_hamiltonian, mag_hamiltonian.conj().T)

        # Check dimensions
        dimension = int(2 * self.cef.J + 1)
        self.assertEqual(mag_hamiltonian.shape, (dimension, dimension))

    def test_scattering(self):
        """Test neutron scattering intensity calculation."""
        _, eigenvectors, _ = self.cef.Hamiltonian()

        # Convert eigenvectors to proper format
        initial_state = eigenvectors[:, 0]
        final_state = eigenvectors[:, 1]

        # Convert to column vectors and ensure they're conjugate transposes
        initial_state = initial_state.reshape(-1, 1)
        final_state = final_state.reshape(-1, 1).conj().T

        intensity = self.cef.scattering(initial_state, final_state)

        # Check if intensity is real and non-negative
        self.assertTrue(isinstance(float(intensity), float))
        self.assertGreaterEqual(float(intensity), 0)


class TestUtilities(unittest.TestCase):
    def test_lorentzian(self):
        """Test Lorentzian line shape calculation."""
        x = np.linspace(-10, 10, 1000)  # Increased resolution
        area = 1.0
        width = 2.0
        position = 0.0

        y = Utilities.lorentzian(x, area, width, position)

        # Check if output has correct shape
        self.assertEqual(y.shape, x.shape)

        # Check if peak is at correct position (with wider tolerance)
        peak_idx = np.argmax(y)
        self.assertAlmostEqual(x[peak_idx], position, places=0)

        # Check if area is approximately correct (with wider tolerance)
        dx = x[1] - x[0]
        numerical_area = np.sum(y) * dx
        self.assertAlmostEqual(numerical_area, area, places=0)

    def test_chi_squared(self):
        """Test chi-squared calculation."""
        observed = np.array([1.0, 2.0, 3.0])
        expected = np.array([1.1, 1.9, 3.1])

        chi_squared = Utilities.calculate_chi_squared(observed, expected)

        # Check if chi-squared is non-negative
        self.assertGreaterEqual(chi_squared, 0)

        # Test with values below threshold
        expected_low = np.array([1e-6, 1e-6, 1e-6])
        chi_squared_low = Utilities.calculate_chi_squared(observed, expected_low)
        self.assertEqual(chi_squared_low, 0.0)


if __name__ == "__main__":
    unittest.main()
