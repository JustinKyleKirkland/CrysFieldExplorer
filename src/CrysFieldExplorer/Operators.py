"""
CrysFieldExplorer - A program for fast optimization of CEF Parameters
Copyright (c) 2023 Kyle Ma
Licensed under the terms of the LICENSE file included in the project.

This module implements the quantum operators and Stevens operators for rare earth ions.
"""

from typing import Dict, List

import numpy as np
import numpy.typing as npt
from numpy import sqrt

# Version and citation information
VERSION = "1.0.0"
CITATION = "J. Appl. Cryst. (2023). 56, 1229-124"
DOI = "https://doi.org/10.1107/S1600576723005897"

# Program header
HEADER = f"""
{"-" * 55}
|                CrysFieldExplorer {VERSION}              |
|   A program for fast optimization of CEF Parameters |
|   -Developed by Kyle Ma                             |
|   Please cite  {CITATION} |
|    {DOI}        |
{"-" * 55}
"""

print(HEADER)

# The [S, L, J] quantum numbers for rare earth elements
# Data sourced from PyCrystalField
RARE_EARTH_IONS: Dict[str, List[float]] = {
    "Ce3+": [0.5, 3.0, 2.5],
    "Pr3+": [1.0, 5.0, 4.0],
    "Nd3+": [1.5, 6.0, 4.5],
    "Pm3+": [2.0, 6.0, 4.0],
    "Sm3+": [2.5, 5.0, 2.5],
    "Eu3+": [3.0, 3.0, 0.0],
    "Gd3+": [3.5, 0.0, 3.5],
    "Tb3+": [3.0, 3.0, 6.0],
    "Dy3+": [2.5, 5.0, 7.5],
    "Ho3+": [2.0, 6.0, 8.0],
    "Er3+": [1.5, 6.0, 7.5],
    "Tm3+": [1.0, 5.0, 6.0],
    "Yb3+": [0.5, 3.0, 3.5],
}


class QuantumOperator:
    """
    Represents the total angular momentum operator for a magnetic ion.

    This class implements various quantum mechanical operators related to angular momentum,
    including raising/lowering operators and the components of the angular momentum vector.

    Args:
        magnetic_ion (str): The magnetic ion symbol (e.g., "Er3+")

    Attributes:
        S (float): The spin quantum number
        L (float): The orbital angular momentum quantum number
        J (float): The total angular momentum quantum number
    """

    def __init__(self, magnetic_ion: str) -> None:
        """Initialize the quantum operator for a given magnetic ion."""
        if magnetic_ion not in RARE_EARTH_IONS:
            raise ValueError(f"Unknown magnetic ion: {magnetic_ion}")

        self.magnetic_ion = magnetic_ion
        self.S = RARE_EARTH_IONS[magnetic_ion][0]
        self.L = RARE_EARTH_IONS[magnetic_ion][1]
        self.J = RARE_EARTH_IONS[magnetic_ion][2]
        self._matrix_size = int(2 * self.J + 1)

    def j_plus(self) -> npt.NDArray:
        """Calculate the angular momentum raising operator J+.

        Returns:
            np.ndarray: The matrix representation of J+
        """
        j_plus = np.zeros((self._matrix_size, self._matrix_size))
        for i in range(self._matrix_size - 1):
            m = -self.J + i
            j_plus[i + 1, i] = sqrt((self.J - m) * (self.J + m + 1))
        return j_plus

    def j_minus(self) -> npt.NDArray:
        """Calculate the angular momentum lowering operator J-.

        Returns:
            np.ndarray: The matrix representation of J-
        """
        j_minus = np.zeros((self._matrix_size, self._matrix_size))
        for i in range(self._matrix_size - 1):
            m = -self.J + i
            j_minus[i, i + 1] = sqrt((self.J + m + 1) * (self.J - m))
        return j_minus

    def j_z(self) -> npt.NDArray:
        """Calculate the z-component of angular momentum Jz.

        Returns:
            np.ndarray: The matrix representation of Jz
        """
        j_z = np.zeros((self._matrix_size, self._matrix_size))
        for i in range(self._matrix_size):
            j_z[i, i] = -self.J + i
        return j_z

    def j_x(self) -> npt.NDArray:
        """Calculate the x-component of angular momentum Jx.

        Returns:
            np.ndarray: The matrix representation of Jx
        """
        return 0.5 * (self.j_plus() + self.j_minus())

    def j_y(self) -> npt.NDArray:
        """Calculate the y-component of angular momentum Jy.

        Returns:
            np.ndarray: The matrix representation of Jy
        """
        return (self.j_minus() - self.j_plus()) / (2j)

    def j_squared(self) -> npt.NDArray:
        """Calculate the total angular momentum operator J².

        Returns:
            np.ndarray: The matrix representation of J²
        """
        j_x, j_y, j_z = self.j_x(), self.j_y(), self.j_z()
        return (
            np.dot(j_x, np.transpose(np.conj(j_x)))
            + np.dot(j_y, np.transpose(np.conj(j_y)))
            + np.dot(j_z, np.transpose(np.conj(j_z)))
        )

    def j_plus_squared(self) -> npt.NDArray:
        """Calculate J+².

        Returns:
            np.ndarray: The matrix representation of J+²
        """
        j_plus = self.j_plus()
        return j_plus @ j_plus

    def j_minus_squared(self) -> npt.NDArray:
        """Calculate J-².

        Returns:
            np.ndarray: The matrix representation of J-²
        """
        j_minus = self.j_minus()
        return j_minus @ j_minus

    def get_matrices(self) -> Dict[str, npt.NDArray]:
        """Get all quantum operator matrices.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing all operator matrices
        """
        return {
            "jx": self.j_x(),
            "jy": self.j_y(),
            "jz": self.j_z(),
            "jplus": self.j_plus(),
            "jminus": self.j_minus(),
            "jsquare": self.j_squared(),
        }


class StevensOperator(QuantumOperator):
    """
    Calculates Stevens Operators for crystal field calculations.

    This class extends QuantumOperator to calculate Stevens Operators, which are used
    in crystal field calculations. The operators are calculated using the quantum
    mechanical operators inherited from the parent class.

    The Stevens Operators are labeled as O_n^m where:
    - n is the order (2, 4, or 6)
    - m is the projection (-n ≤ m ≤ n)

    Args:
        magnetic_ion (str): The magnetic ion symbol (e.g., "Er3+")
    """

    def __init__(self, magnetic_ion: str) -> None:
        """Initialize the Stevens Operator calculator."""
        super().__init__(magnetic_ion)
        self._operators_cache = {}

        # Define mapping of (n,m) to calculation methods
        self._operator_map = {
            # n = 2 operators
            (2, 0): lambda jx, jy, jz, jp, jm, js: self._calculate_o20(jz, js),
            (2, 1): lambda jx, jy, jz, jp, jm, js: self._calculate_o21(jz, jp, jm),
            (2, 2): lambda jx, jy, jz, jp, jm, js: self._calculate_o22(jp, jm),
            (2, -2): lambda jx, jy, jz, jp, jm, js: self._calculate_o2n2(jp, jm),
            # n = 4 operators
            (4, 0): lambda jx, jy, jz, jp, jm, js: self._calculate_o40(jz, js),
            (4, 1): lambda jx, jy, jz, jp, jm, js: self._calculate_o41(jz, jp, jm, js),
            (4, 2): lambda jx, jy, jz, jp, jm, js: self._calculate_o42(jz, jp, jm, js),
            (4, 3): lambda jx, jy, jz, jp, jm, js: self._calculate_o43(jz, jp, jm),
            (4, 4): lambda jx, jy, jz, jp, jm, js: self._calculate_o44(jp, jm),
            (4, -2): lambda jx, jy, jz, jp, jm, js: self._calculate_o4n2(jz, jp, jm, js),
            (4, -3): lambda jx, jy, jz, jp, jm, js: self._calculate_o4n3(jz, jp, jm),
            (4, -4): lambda jx, jy, jz, jp, jm, js: self._calculate_o4n4(jp, jm),
            # n = 6 operators
            (6, 0): lambda jx, jy, jz, jp, jm, js: self._calculate_o60(jz, js),
            (6, 1): lambda jx, jy, jz, jp, jm, js: self._calculate_o61(jz, jp, jm, js),
            (6, 2): lambda jx, jy, jz, jp, jm, js: self._calculate_o62(jz, jp, jm, js),
            (6, 3): lambda jx, jy, jz, jp, jm, js: self._calculate_o63(jz, jp, jm, js),
            (6, 4): lambda jx, jy, jz, jp, jm, js: self._calculate_o64(jz, jp, jm, js),
            (6, 5): lambda jx, jy, jz, jp, jm, js: self._calculate_o65(jz, jp, jm),
            (6, 6): lambda jx, jy, jz, jp, jm, js: self._calculate_o66(jp, jm),
            (6, -2): lambda jx, jy, jz, jp, jm, js: self._calculate_o6n2(jz, jp, jm, js),
            (6, -3): lambda jx, jy, jz, jp, jm, js: self._calculate_o6n3(jz, jp, jm, js),
            (6, -4): lambda jx, jy, jz, jp, jm, js: self._calculate_o6n4(jz, jp, jm, js),
            (6, -6): lambda jx, jy, jz, jp, jm, js: self._calculate_o6n6(jp, jm),
        }

    def _calculate_o20(self, jz: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_2^0 operator."""
        return 3.0 * (jz**2) - jsquare

    def _calculate_o21(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_2^1 operator."""
        sum_ops = jplus + jminus
        return 0.25 * (jz @ sum_ops + sum_ops @ jz)

    def _calculate_o22(self, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_2^2 operator."""
        return 0.5 * (jplus**2 + jminus**2)

    def _calculate_o2n2(self, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_2^-2 operator."""
        return (-0.5j) * (jplus**2 - jminus**2)

    def _calculate_o40(self, jz: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_4^0 operator."""
        return 35.0 * (jz**4) - 30.0 * (jsquare @ jz**2) + 25.0 * (jz**2) - 6.0 * jsquare + 3.0 * (jsquare**2)

    def _calculate_o41(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_4^1 operator."""
        sum_ops = jplus + jminus
        term = 7 * jz**3 - (3 * jsquare @ jz + jz)
        return 0.25 * (sum_ops @ term + term @ sum_ops)

    def _calculate_o42(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_4^2 operator."""
        sum_squared = jplus**2 + jminus**2
        term = 7 * jz**2 - jsquare
        return 0.25 * (sum_squared @ term - 5 * sum_squared + term @ sum_squared - 5 * sum_squared)

    def _calculate_o43(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_4^3 operator."""
        sum_cubed = jplus**3 + jminus**3
        return 0.25 * (jz @ sum_cubed + sum_cubed @ jz)

    def _calculate_o44(self, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_4^4 operator."""
        return 0.5 * (jplus**4 + jminus**4)

    def _calculate_o4n2(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_4^-2 operator."""
        diff_squared = jplus**2 - jminus**2
        term = 7 * jz**2 - jsquare
        return (-0.25j) * (diff_squared @ term - 5 * diff_squared + term @ diff_squared - 5 * diff_squared)

    def _calculate_o4n3(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_4^-3 operator."""
        diff_cubed = jplus**3 - jminus**3
        return (-0.25j) * (diff_cubed @ jz + jz @ diff_cubed)

    def _calculate_o4n4(self, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_4^-4 operator."""
        return (-0.5j) * (jplus**4 - jminus**4)

    def _calculate_o60(self, jz: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_6^0 operator."""
        return (
            231 * (jz**6)
            - 315 * (jsquare @ jz**4)
            + 735 * (jz**4)
            + 105 * (jsquare**2 @ jz**2)
            - 525 * (jsquare @ jz**2)
            + 294 * (jz**2)
            - 5 * (jsquare**3)
            + 40 * (jsquare**2)
            - 60 * jsquare
        )

    def _calculate_o61(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_6^1 operator."""
        sum_ops = jplus + jminus
        term = 33 * jz**5 - (30 * jsquare @ jz**3 - 15 * jz**3) + (5 * jsquare**2 @ jz - 10 * jsquare @ jz + 12 * jz)
        return 0.25 * (sum_ops @ term + term @ sum_ops)

    def _calculate_o62(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_6^2 operator."""
        sum_squared = jplus**2 + jminus**2
        term = 33 * jz**4 - (18 * jsquare @ jz**2 + 123 * jz**2) + jsquare**2 + 10 * jsquare
        return 0.25 * (sum_squared @ term + 102 * sum_squared + term @ sum_squared + 102 * sum_squared)

    def _calculate_o63(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_6^3 operator."""
        sum_cubed = jplus**3 + jminus**3
        term = 11 * jz**3 - 3 * jsquare @ jz - 59 * jz
        return 0.25 * (term @ sum_cubed + sum_cubed @ term)

    def _calculate_o64(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_6^4 operator."""
        sum_fourth = jplus**4 + jminus**4
        term = 11 * jz**2 - jsquare
        return 0.25 * (sum_fourth @ term - 38 * sum_fourth + term @ sum_fourth - 38 * sum_fourth)

    def _calculate_o65(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_6^5 operator."""
        sum_fifth = jplus**5 + jminus**5
        return 0.25 * (sum_fifth @ jz + jz @ sum_fifth)

    def _calculate_o66(self, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_6^6 operator."""
        return 0.5 * (jplus**6 + jminus**6)

    def _calculate_o6n2(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_6^-2 operator."""
        diff_squared = jplus**2 - jminus**2
        term = 33 * jz**4 - (18 * jsquare @ jz**2 + 123 * jz**2) + jsquare**2 + 10 * jsquare
        return (-0.25j) * (diff_squared @ term + 102 * diff_squared + term @ diff_squared + 102 * diff_squared)

    def _calculate_o6n3(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_6^-3 operator."""
        diff_cubed = jplus**3 - jminus**3
        term = 11 * jz**3 - 3 * jsquare @ jz - 59 * jz
        return (-0.25j) * (term @ diff_cubed + diff_cubed @ term)

    def _calculate_o6n4(self, jz: np.ndarray, jplus: np.ndarray, jminus: np.ndarray, jsquare: np.ndarray) -> np.ndarray:
        """Calculate O_6^-4 operator."""
        diff_fourth = jplus**4 - jminus**4
        term = 11 * jz**2 - jsquare
        return (-0.25j) * (diff_fourth @ term - 38 * diff_fourth + term @ diff_fourth - 38 * diff_fourth)

    def _calculate_o6n6(self, jplus: np.ndarray, jminus: np.ndarray) -> np.ndarray:
        """Calculate O_6^-6 operator."""
        return (-0.5j) * (jplus**6 - jminus**6)

    def get_operator(self, n: int, m: int) -> np.ndarray:
        """
        Get the Stevens Operator O_n^m.

        Args:
            n (int): Order of the operator (2, 4, or 6)
            m (int): Projection (-n ≤ m ≤ n)

        Returns:
            np.ndarray: Matrix representation of the Stevens Operator

        Raises:
            ValueError: If n,m combination is invalid
            NotImplementedError: If the requested operator is not implemented
        """
        if not (n in [2, 4, 6] and -n <= m <= n):
            raise ValueError(f"Invalid Stevens Operator indices: n={n}, m={m}")

        # Check cache first
        cache_key = f"{n}{m}"
        if cache_key in self._operators_cache:
            return self._operators_cache[cache_key]

        # Get base operators
        matrices = super().get_matrices()
        jx = np.array(matrices["jx"])
        jy = np.array(matrices["jy"])
        jz = np.array(matrices["jz"])
        jplus = np.array(matrices["jplus"])
        jminus = np.array(matrices["jminus"])
        jsquare = np.array(matrices["jsquare"])

        # Get the calculation function from the map
        calc_func = self._operator_map.get((n, m))
        if calc_func is None:
            # For negative m values not explicitly defined, use the relation to positive m
            if m < 0 and (n, abs(m)) in self._operator_map:
                pos_m_op = self.get_operator(n, abs(m))
                if abs(m) == 1:
                    # For m=±1, O_n^-1 = i * O_n^1
                    matrix = 1j * np.array(pos_m_op)
                else:
                    # For |m|≥2, need to handle each element's phase separately
                    # This is a placeholder until we understand the full pattern
                    matrix = 1j * np.array(pos_m_op)
                    # Adjust signs based on matrix position if needed
                    if abs(m) == 2:
                        # For m=±2, some elements need sign flips
                        matrix = self._adjust_phase_m2(matrix)
            else:
                raise NotImplementedError(f"Stevens Operator O_{n}^{m} not yet implemented")
        else:
            matrix = calc_func(jx, jy, jz, jplus, jminus, jsquare)

        # Ensure the result is a numpy array
        matrix = np.array(matrix)

        # Cache the result
        self._operators_cache[cache_key] = matrix
        return matrix

    def _adjust_phase_m2(self, matrix: np.ndarray) -> np.ndarray:
        """Adjust phases for m=±2 operators based on matrix position."""
        # Create a copy to avoid modifying the input
        result = matrix.copy()
        # Flip signs of appropriate elements
        # This is a placeholder - we need to determine the exact pattern
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i > j:  # Lower triangle
                    result[i, j] = -matrix[i, j]
        return result

    def get_all_operators(self) -> Dict[str, np.ndarray]:
        """
        Calculate all implemented Stevens Operators.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping operator labels to matrices
        """
        operators = {}
        for n in [2, 4, 6]:
            for m in range(-n, n + 1):
                try:
                    operators[f"O_{n}^{m}"] = self.get_operator(n, m)
                except NotImplementedError:
                    continue
        return operators
