"""CrysFieldExplorer main module for crystal field calculations.

This module provides the core functionality for crystal field calculations,
including Hamiltonian construction, intensity calculations, and various
physical property computations.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from alive_progress import alive_bar
from scipy import linalg
from sympy import Matrix, cos, sin, solve, symbols

import CrysFieldExplorer.Operators as op
import CrysFieldExplorer.Visulization as vis

# Physical constants in SI units
KB_MEV_K = 8.61733e-2  # Boltzmann constant in meV/K
NA = 6.0221409e23  # Avogadro number
MU_B = 9.274009994e-21  # Bohr magneton
MU_B_TESLA = 5.7883818012e-2  # Bohr magneton in Tesla

ArrayLike = Union[np.ndarray, List[float]]


class CrysFieldExplorer(op.Stevens_Operator, op.Quantum_Operator):
    """Crystal Field Explorer class for calculating crystal field properties.

    This class inherits from Stevens_Operator and Quantum_Operator to provide
    access to all quantum operators needed for crystal field calculations.

    Attributes:
        stevens_idx: List of [n,m] pairs for Stevens Operator indices
        alpha: Radial portion of CEF wave function (2nd order)
        beta: Radial portion of CEF wave function (4th order)
        gamma: Radial portion of CEF wave function (6th order)
        parameters: Crystal field parameters
        temperature: Temperature in Kelvin
        field: External magnetic field vector [Bx,By,Bz]

    Args:
        magnetic_ion: Name of magnetic ion (e.g. "Er3+")
        stevens_idx: List of [n,m] pairs where n=subscript, m=superscript
        alpha: Radial portion coefficient (2nd order)
        beta: Radial portion coefficient (4th order)
        gamma: Radial portion coefficient (6th order)
        parameters: Crystal field parameters
        temperature: Temperature in Kelvin
        field: External magnetic field vector [Bx,By,Bz]
    """

    def __init__(
        self,
        magnetic_ion: str,
        stevens_idx: List[List[int]],
        alpha: float,
        beta: float,
        gamma: float,
        parameters: Union[Dict[int, float], List[float], np.ndarray],
        temperature: float,
        field: List[float],
    ) -> None:
        self.stevens_idx = stevens_idx
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.parameters = parameters if isinstance(parameters, dict) else {i: p for i, p in enumerate(parameters)}
        self.temperature = temperature
        self.field = field
        super().__init__(magnetic_ion)

    def Hamiltonian(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the crystal field Hamiltonian and its eigenvalues/vectors.

        Constructs the Hamiltonian using Stevens operators and calculates
        its eigenvalues and eigenvectors.

        Returns:
            Tuple containing:
            - eigenvalues: Sorted eigenvalues of the Hamiltonian
            - eigenvectors: Corresponding eigenvectors
            - hamiltonian: The Hamiltonian matrix
        """
        operators = super().Stevens_hash(self.stevens_idx)
        hamiltonian = np.zeros_like(operators[list(operators.keys())[0]], dtype=complex)

        # Map order to coefficient for faster lookup
        coeff_map = {"2": self.alpha, "4": self.beta, "6": self.gamma}

        for j, (key, operator) in enumerate(operators.items()):
            order = key[0]  # First character is the order (2,4,6)
            hamiltonian += coeff_map[order] * self.parameters[j] * operator

        eigenvalues, eigenvectors = linalg.eigh(hamiltonian)
        sort_idx = np.argsort(eigenvalues)

        return eigenvalues[sort_idx], eigenvectors[:, sort_idx], hamiltonian

    @classmethod
    def Hamiltonian_scale(
        cls,
        magnetic_ion: str,
        stevens_idx: List[List[int]],
        alpha: float,
        beta: float,
        gamma: float,
        parameters: Union[Dict[int, float], List[float], np.ndarray],
        scale: List[float],
        temperature: float,
        field: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate scaled Hamiltonian for fitting purposes.

        Args:
            magnetic_ion: Name of the magnetic ion
            stevens_idx: Stevens operator indices
            alpha: Stevens alpha parameter
            beta: Stevens beta parameter
            gamma: Stevens gamma parameter
            parameters: Crystal field parameters
            scale: Scaling factors for operators
            temperature: Temperature in Kelvin
            field: Applied magnetic field vector

        Returns:
            Tuple of (eigenvalues, eigenvectors, hamiltonian)
        """
        instance = cls(magnetic_ion, stevens_idx, alpha, beta, gamma, parameters, temperature, field)
        operators = instance.Stevens_hash(stevens_idx)

        # Scale operators
        for idx, (key, operator) in enumerate(operators.items()):
            operators[key] = operator * scale[idx]

        # Build Hamiltonian with scaled operators
        hamiltonian = np.zeros_like(operators[list(operators.keys())[0]], dtype=complex)
        coeff_map = {"2": alpha, "4": beta, "6": gamma}

        for idx, (key, operator) in enumerate(operators.items()):
            order = key[0]
            hamiltonian += coeff_map[order] * parameters[idx] * operator

        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        sort_idx = np.argsort(eigenvalues)

        return eigenvalues[sort_idx], eigenvectors[:, sort_idx], hamiltonian

    def calculate_lande_g_factor(self) -> float:
        """Calculate the Landé g-factor.

        Returns:
            float: The calculated Landé g-factor
        """
        S, L, J = self.S, self.L, self.J
        return 1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))

    def magnetic_Hamiltonian(self, Bx: float, By: float, Bz: float) -> np.ndarray:
        """Calculate the magnetic field Hamiltonian.

        Computes the magnetic field contribution to the Hamiltonian using
        the Zeeman interaction term: -gJ μB J·B

        Args:
            Bx: Magnetic field component along x-axis (Tesla)
            By: Magnetic field component along y-axis (Tesla)
            Bz: Magnetic field component along z-axis (Tesla)

        Returns:
            The magnetic field Hamiltonian matrix
        """
        # Get angular momentum operators
        jx, jy, jz = super().Jx(), super().Jy(), super().Jz()

        # Calculate Landé g-factor
        g_factor = self.calculate_lande_g_factor()

        # Construct magnetic Hamiltonian
        return -g_factor * MU_B_TESLA * (Bx * jx + By * jy + Bz * jz)

    def solve_magnetic_system(self, Bx: float, By: float, Bz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the system with magnetic field contribution.

        Args:
            Bx: Magnetic field component along x-axis (Tesla)
            By: Magnetic field component along y-axis (Tesla)
            Bz: Magnetic field component along z-axis (Tesla)

        Returns:
            Tuple of (energy_levels, eigenvectors, magnetic_hamiltonian)
        """
        _, _, crystal_field_hamiltonian = self.Hamiltonian()
        magnetic_hamiltonian = self.magnetic_Hamiltonian(Bx, By, Bz)

        total_hamiltonian = crystal_field_hamiltonian + magnetic_hamiltonian
        eigenvalues, eigenvectors = np.linalg.eigh(total_hamiltonian)

        # Reference energies to ground state
        energy_levels = eigenvalues - eigenvalues[0]

        return energy_levels, eigenvectors, magnetic_hamiltonian

    def magsovler(self, Bx, By, Bz):
        _, _, H = self.Hamiltonian()
        magH = self.magnetic_Hamiltonian(Bx, By, Bz)
        Eigenvalues, Eigenvectors = np.linalg.eigh(H + magH)
        Energy = Eigenvalues - Eigenvalues[0]
        return Energy, Eigenvectors, magH

    def scattering(self, initial_state: np.ndarray, final_state: np.ndarray) -> float:
        """Calculate the neutron scattering intensity between two states.

        Computes the magnetic neutron scattering intensity between states using
        the dipole approximation.

        Args:
            initial_state: Initial state vector
            final_state: Final state vector

        Returns:
            float: Scattering intensity between states
        """
        # Get angular momentum operators
        jx, jy, jz = super().Jx(), super().Jy(), super().Jz()

        # Calculate matrix elements for each direction
        matrix_elements = [np.dot(np.dot(initial_state, op), final_state) for op in (jx, jy, jz)]

        # Sum over all directions
        intensity = sum(np.dot(np.conj(elem), elem) for elem in matrix_elements)
        return float(intensity.real)

    def calculate_boltzmann_factor(self, energy_levels: ArrayLike, target_energy: float) -> float:
        """Calculate the temperature-dependent Boltzmann factor.

        Computes the thermal population factor for a given energy level
        at the system temperature.

        Args:
            energy_levels: Array of energy levels
            target_energy: Energy of the level of interest

        Returns:
            float: Boltzmann factor for the given energy level
        """
        beta = 1 / (KB_MEV_K * self.temperature)
        partition_function = np.sum(np.exp(-beta * np.array(energy_levels)))
        return float(np.exp(-beta * target_energy) / partition_function)

    def calculate_transition_intensities(self, ground_state_idx: int) -> Dict[int, float]:
        """Calculate transition intensities from a ground state.

        Computes the neutron scattering intensities for transitions from
        a given ground state to all excited states.

        Args:
            ground_state_idx: Index of the ground state

        Returns:
            Dict[int, float]: Mapping of excited state indices to transition intensities
        """
        eigenvalues, eigenvectors, _ = self.Hamiltonian()
        energies = eigenvalues - eigenvalues[0]  # Reference to ground state

        # Calculate Boltzmann factor once for ground state
        boltzmann_factor = self.calculate_boltzmann_factor(energies, energies[ground_state_idx])

        # Calculate intensities for transitions to all excited states
        intensities = {}
        ground_state = eigenvectors[:, ground_state_idx].H

        for excited_idx in range(ground_state_idx, int(2 * self.J + 1)):
            intensity = boltzmann_factor * self.scattering(ground_state, eigenvectors[:, excited_idx])
            intensities[excited_idx] = intensity

        return intensities

    def calculate_normalized_intensities(
        self, reference_idx: int, ground_state_idx: int, consider_kramers: bool = True
    ) -> Dict[int, float]:
        """Calculate normalized neutron scattering intensities.

        Computes the neutron scattering intensities normalized to a reference
        transition, taking into account Kramers degeneracy if applicable.

        Args:
            reference_idx: Index of the reference transition for normalization
            ground_state_idx: Index of the ground state
            consider_kramers: Whether to account for Kramers degeneracy

        Returns:
            Dict[int, float]: Mapping of state indices to normalized intensities
        """
        intensities = self.calculate_transition_intensities(ground_state_idx)

        if consider_kramers:
            # Sum over Kramers doublets
            summed_intensities = {}
            for i in range(0, len(intensities), 2):
                summed_intensities[i] = intensities[i] + intensities[i + 1]
        else:
            summed_intensities = intensities

        # Normalize to reference transition
        reference_intensity = summed_intensities[reference_idx]
        return {idx: (intensity / reference_intensity).item() for idx, intensity in summed_intensities.items()}

    def calculate_intensities_fast(self, ground_state_idx: int) -> np.ndarray:
        """Calculate transition intensities using vectorized operations.

        A faster version of calculate_transition_intensities() that uses numpy arrays
        instead of dictionaries for better performance.

        Args:
            ground_state_idx: Index of the ground state

        Returns:
            np.ndarray: Array of transition intensities from ground state to all states
        """
        eigenvalues, eigenvectors, _ = self.Hamiltonian()
        energies = eigenvalues - eigenvalues[0]

        # Calculate Boltzmann factor once
        boltzmann_factor = self.calculate_boltzmann_factor(energies, energies[ground_state_idx])
        ground_state = eigenvectors[:, ground_state_idx].H

        # Calculate all intensities at once
        intensities = [
            boltzmann_factor * self.scattering(ground_state, eigenvectors[:, i])
            for i in range(ground_state_idx, int(2 * self.J + 1))
        ]

        return np.array(intensities).squeeze()

    def calculate_normalized_intensities_fast(self, reference_idx: int, ground_state_idx: int) -> np.ndarray:
        """Fast calculation of normalized neutron intensities for Kramers ions.

        A vectorized version of calculate_normalized_intensities() specifically for
        Kramers ions where states come in degenerate pairs.

        Args:
            reference_idx: Index of the reference transition for normalization
            ground_state_idx: Index of the ground state

        Returns:
            np.ndarray: Array of normalized intensities
        """
        intensities = self.calculate_intensities_fast(ground_state_idx)

        # Sum over Kramers doublets
        summed_intensities = [(intensities[i] + intensities[i + 1]).real for i in range(0, len(intensities), 2)]

        # Normalize to reference transition
        summed_intensities = np.array(summed_intensities)
        return summed_intensities / summed_intensities[reference_idx]

    def calculate_magnetic_intensities(self, ground_state_idx: int) -> np.ndarray:
        """Calculate transition intensities in presence of magnetic field.

        Computes scattering intensities between states split by both crystal
        field and magnetic field effects.

        Args:
            ground_state_idx: Index of the ground state

        Returns:
            np.ndarray: Array of transition intensities from ground state to all states
        """
        eigenvalues, eigenvectors, _ = self.solve_magnetic_system(*self.field)
        energies = eigenvalues - eigenvalues[0]

        # Calculate Boltzmann factor once
        boltzmann_factor = self.calculate_boltzmann_factor(energies, energies[ground_state_idx])
        ground_state = eigenvectors[:, ground_state_idx].H

        # Calculate all intensities at once
        intensities = [
            boltzmann_factor * self.scattering(ground_state, eigenvectors[:, i])
            for i in range(ground_state_idx, int(2 * self.J + 1))
        ]

        return np.array(intensities).squeeze()

    def calculate_normalized_magnetic_intensities(self, reference_idx: int, ground_state_idx: int) -> np.ndarray:
        """Calculate normalized neutron intensities with magnetic field.

        Computes normalized scattering intensities between states split by both
        crystal field and magnetic field effects. Note that energy levels
        are now split due to Zeeman effect.

        Args:
            reference_idx: Index of the reference transition for normalization
            ground_state_idx: Index of the ground state

        Returns:
            np.ndarray: Array of normalized intensities
        """
        intensities = self.calculate_magnetic_intensities(ground_state_idx)

        # Sum over pairs of states (may not be degenerate due to field)
        summed_intensities = [(intensities[i] + intensities[i + 1]).real for i in range(0, len(intensities), 2)]

        # Normalize to reference transition
        summed_intensities = np.array(summed_intensities)
        return summed_intensities / summed_intensities[reference_idx]

    def calculate_g_tensor(self) -> np.ndarray:
        """Calculate the g-tensor for the ground state doublet.

        Returns:
            np.ndarray: The 3x3 g-tensor matrix
        """
        # Get angular momentum operators
        jx, jy, jz = super().Jx(), super().Jy(), super().Jz()

        # Calculate g-factor
        g_factor = self.calculate_lande_g_factor()

        # Get ground state doublet wavefunctions
        _, eigenvectors, _ = self.Hamiltonian()
        plus = eigenvectors[:, 0]  # Ground state
        minus = eigenvectors[:, 1]  # First excited state

        # Calculate matrix elements
        matrix_elements = {
            "x": self._calculate_matrix_elements(jx, plus, minus),
            "y": self._calculate_matrix_elements(jy, plus, minus),
            "z": self._calculate_matrix_elements(jz, plus, minus),
        }

        # Construct g-tensor
        g_tensor = np.array(
            [
                [matrix_elements["x"][1][0].real, matrix_elements["x"][1][0].imag, matrix_elements["x"][0][0].real],
                [matrix_elements["y"][1][0].real, matrix_elements["y"][1][0].imag, matrix_elements["y"][0][0].real],
                [matrix_elements["z"][1][0].real, matrix_elements["z"][1][0].imag, matrix_elements["z"][0][0].real],
            ]
        )

        return 2 * g_factor * g_tensor

    def _calculate_matrix_elements(
        self, operator: np.ndarray, plus: np.ndarray, minus: np.ndarray
    ) -> List[List[complex]]:
        """Calculate matrix elements for g-tensor calculation.

        Args:
            operator: Angular momentum operator (Jx, Jy, or Jz)
            plus: Ground state wavefunction
            minus: First excited state wavefunction

        Returns:
            List[List[complex]]: 2x2 matrix of operator elements
        """
        return [
            [(plus.H * operator * plus).item(), (minus.H * operator * plus).item()],
            [(plus.H * operator * minus).item(), (minus.H * operator * minus).item()],
        ]

    def calculate_rotated_g_tensor(self, axis: str) -> Tuple[float, float, float, np.ndarray]:
        """Calculate the rotated g-tensor to diagonalize specific components.

        Args:
            axis: Rotation axis ('x', 'y', or 'z')

        Returns:
            Tuple containing:
            - gxx: g-tensor xx component
            - gyy: g-tensor yy component
            - gzz: g-tensor zz component
            - rotated_g_tensor: The rotated g-tensor matrix
        """
        g_tensor = self.calculate_g_tensor()
        theta = symbols("theta")

        # Define rotation matrices
        rotation_matrices = {
            "x": Matrix([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]]),
            "y": Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]),
            "z": Matrix([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]),
        }

        if axis not in rotation_matrices:
            raise ValueError(f"Invalid axis: {axis}. Must be one of: x, y, z")

        # Rotate g-tensor
        rotation = rotation_matrices[axis]
        rotated = g_tensor * rotation.inv()

        # Find angle that makes off-diagonal elements zero
        theta_value = solve(rotated[0, 2] - rotated[2, 0])[0]

        # Apply rotation with solved angle
        if axis == "y":  # Most common case
            final_rotation = Matrix(
                [[cos(theta_value), 0, sin(theta_value)], [0, 1, 0], [-sin(theta_value), 0, cos(theta_value)]]
            )
        else:
            final_rotation = rotation.subs(theta, theta_value)

        rotated_g_tensor = np.array(g_tensor * final_rotation.inv(), dtype=float)

        # Extract principal values
        eigenvalues = np.abs(np.linalg.eigvals(rotated_g_tensor))
        gxx, gyy, gzz = eigenvalues[0], eigenvalues[2], eigenvalues[1]

        return gxx, gyy, gzz, rotated_g_tensor


class Utilities(CrysFieldExplorer):
    """Utility functions for crystal field calculations.

    This class contains functions to calculate loss functions, construct neutron spectra,
    magnetization, susceptibility and other common functions needed in CrysFieldExplorer
    for both optimization and visualization.
    """

    def __init__(
        self,
        magnetic_ion: str,
        stevens_idx: List[List[int]],
        alpha: float,
        beta: float,
        gamma: float,
        parameters: Union[Dict[int, float], List[float], np.ndarray],
        temperature: float,
        field: List[float],
    ):
        super().__init__(magnetic_ion, stevens_idx, alpha, beta, gamma, parameters, temperature, field)

    @staticmethod
    def lorentzian(x: np.ndarray, area: float, width: float, position: float) -> np.ndarray:
        """Calculate Lorentzian line shape.

        Args:
            x: x-axis values
            area: Area under the curve
            width: Full width at half maximum
            position: Peak position

        Returns:
            np.ndarray: Lorentzian line shape values
        """
        return (area / np.pi) * (width / 2) / ((x - position) ** 2 + (width / 2) ** 2)

    @staticmethod
    def calculate_chi_squared(observed: np.ndarray, expected: np.ndarray, threshold: float = 2e-5) -> float:
        """Calculate chi-squared statistic.

        Args:
            observed: Observed values
            expected: Expected values
            threshold: Minimum expected value to consider

        Returns:
            float: Chi-squared statistic
        """
        valid_indices = expected >= threshold
        if not np.any(valid_indices):
            return 0.0

        diff_squared = (observed[valid_indices] - expected[valid_indices]) ** 2
        return float(np.sum(diff_squared / expected[valid_indices]))

    def calculate_differential_susceptibility(self, temperature: float, sampling: int = 1000) -> float:
        """Calculate magnetic susceptibility using differential method.

        Calculates susceptibility by taking differential of dm/dh using Monte Carlo sampling.

        Args:
            temperature: Temperature in Kelvin
            sampling: Number of Monte Carlo samples

        Returns:
            float: Magnetic susceptibility
        """
        # Small field for numerical derivative
        field_magnitude = 0.01  # Tesla

        # Calculate g-factor once
        g_factor = self.calculate_lande_g_factor()

        total_magnetization = 0.0
        for _ in range(sampling):
            # Generate random direction
            direction = np.random.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction)

            # Apply field in random direction
            field = field_magnitude * direction
            energies, eigenvectors, _ = self.solve_magnetic_system(*field)

            # Calculate partition function
            beta = 1 / (KB_MEV_K * temperature)
            partition_function = np.sum(np.exp(-beta * energies))

            # Calculate magnetization components
            magnetization = np.zeros(3)
            for i, operator in enumerate([self.Jx(), self.Jy(), self.Jz()]):
                for n in range(len(energies)):
                    magnetization[i] += (
                        (eigenvectors[:, n].H * (g_factor * operator) * eigenvectors[:, n])[0, 0].real
                        * np.exp(-beta * energies[n])
                        / partition_function
                    )

            # Project magnetization onto field direction
            total_magnetization += np.dot(magnetization, direction)

        # Convert to appropriate units and average
        return NA * MU_B * total_magnetization / (field_magnitude * sampling)

    def calculate_magnetization_curve(
        self, temperature: float, field_range: Tuple[float, float, float] = (0.1, 7.1, 0.5), sampling: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate powder-averaged magnetization curve.

        Args:
            temperature: Temperature in Kelvin
            field_range: Tuple of (start, end, step) in Tesla
            sampling: Number of Monte Carlo samples per field point

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of field values and corresponding magnetizations
        """
        fields = np.arange(*field_range)
        magnetizations = []

        g_factor = self.calculate_lande_g_factor()
        dimension = int(2 * self.J + 1)

        with alive_bar(len(fields), bar="bubbles") as progress:
            for field_magnitude in fields:
                total_magnetization = 0.0

                for _ in range(sampling):
                    # Generate random direction
                    direction = np.random.normal(0, 1, 3)
                    direction = direction / np.linalg.norm(direction)

                    # Apply field in random direction
                    field = field_magnitude * direction
                    energies, eigenvectors, _ = self.solve_magnetic_system(*field)

                    # Calculate partition function
                    beta = 1 / (KB_MEV_K * temperature)
                    partition_function = np.sum(np.exp(-beta * energies))

                    # Calculate magnetization components
                    magnetization = np.zeros(3)
                    for i, operator in enumerate([self.Jx(), self.Jy(), self.Jz()]):
                        for n in range(dimension):
                            magnetization[i] += (
                                (eigenvectors[:, n].H * (g_factor * operator) * eigenvectors[:, n])[0, 0].real
                                * np.exp(-beta * energies[n])
                                / partition_function
                            )

                    # Project magnetization onto field direction
                    total_magnetization += np.dot(magnetization, direction)

                magnetizations.append(total_magnetization / sampling)
                progress()
                progress.title = "Calculating Magnetization"

        return np.array(fields), np.array(magnetizations)

    def calculate_van_vleck_susceptibility(self, temperature_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate magnetic susceptibility using Van Vleck formula.

        Args:
            temperature_range: Array of temperatures to calculate susceptibility at

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of temperatures and corresponding susceptibilities
        """
        eigenvalues, eigenvectors, _ = self.Hamiltonian()
        energies = eigenvalues - eigenvalues[0]

        g_factor = self.calculate_lande_g_factor()
        dimension = int(2 * self.J + 1)

        # Constants for susceptibility calculation
        prefactor = (g_factor**2) * NA * (MU_B**2) / (1.38064852e-16)  # CGS units

        susceptibilities = []
        for temp in temperature_range:
            # Calculate partition function
            beta = 1 / (KB_MEV_K * temp)
            partition_function = np.sum(np.exp(-beta * energies))

            # Calculate susceptibility terms
            chi = 0.0
            operators = [self.Jx(), self.Jy(), self.Jz()]

            for operator in operators:
                for n in range(dimension):
                    for m in range(dimension):
                        matrix_element = np.abs(eigenvectors[:, m].H * operator * eigenvectors[:, n])[0, 0]

                        if np.abs(energies[m] - energies[n]) < 1e-5:
                            # Diagonal terms
                            chi += matrix_element**2 * np.exp(-beta * energies[n]) / temp
                        else:
                            # Off-diagonal terms
                            chi += 2 * (matrix_element**2) * np.exp(-beta * energies[n]) / (energies[m] - energies[n])

            susceptibilities.append(prefactor * chi / (3 * partition_function))

        return temperature_range, np.array(susceptibilities)


# %% test
if __name__ == "__main__":
    alpha = 0.01 * 10.0 * 4 / (45 * 35)
    beta = 0.01 * 100.0 * 2 / (11 * 15 * 273)
    gamma = 0.01 * 10.0 * 8 / (13**2 * 11**2 * 3**3 * 7)
    Stevens_idx = [
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
    scale = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    test = pd.read_csv(
        "C:/Users/qmc/OneDrive/ONRL/Data/CEF/Python/Eradam/Eradam_MPI_Newfit_goodsolution.csv",
        header=None,
    )
    Parameter = dict()
    temperature = 5
    field = [0, 0, 0]
    j = 0

    Para = np.zeros(15)
    for i in range(15):
        Para[i] = test[i][0]

    Para[2] = 10 * Para[2]  # 22 -2
    Para[4] = 0.1 * Para[4]  # 41 - 4
    Para[6] = 10 * Para[6]  # 43 -6
    Para[9] = 0.1 * Para[9]  # 61 -9
    Para[11] = 10 * Para[11]  # 63 -11
    Para[13] = 10 * Para[13]  # 65 -13
    Para[14] = 10 * Para[14]  # 66 -14

    CEF = CrysFieldExplorer("Er3+", Stevens_idx, alpha, beta, gamma, Para, temperature, field)
    ev, ef, H = CEF.Hamiltonian()
    # classmethod here doesn't associate Hamiltonian_scale to any instance so we doesn't need to create another instance to change scale
    # ev,ef,H=CrysFieldExplorer.Hamiltonian_scale('Er3+', Stevens_idx, alpha, beta, gamma, Para, scale, temperature, field)
    print(np.round(ev - ev[0], 3))

    # Intensity=CEF.Neutron_Intensity(2, 0, True)
    Intensity = CEF.calculate_intensities_fast(0)
    # print(Intensity)

    uti = Utilities("Er3+", Stevens_idx, alpha, beta, gamma, Para, temperature, field)
    ev1, _, _ = uti.test(1, 3, 1)
    print(np.round(ev1 - ev1[0], 3))
    # plotting
    plot = vis.vis(15, 10)
    plot.susceptibility(uti.susceptibility_VanVleck())

    mag = uti.magnetization(5)
    plot.magnetization(mag)

    plot.neutron_spectrum(ev - ev[0], Intensity, 0.5)
