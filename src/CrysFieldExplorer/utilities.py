"""Utilities module for CrysFieldExplorer.

This module provides utility functions for crystal field calculations,
including Lorentzian line shape calculations and other common functions.
"""

from typing import List, Union

import numpy as np

# Physical constants in SI units
KB_MEV_K = 8.61733e-2  # Boltzmann constant in meV/K
NA = 6.0221409e23  # Avogadro number
MU_B = 9.274009994e-21  # Bohr magneton
MU_B_TESLA = 5.7883818012e-2  # Bohr magneton in Tesla

ArrayLike = Union[np.ndarray, List[float]]


class Utilities:
    """Utility functions for crystal field calculations."""

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
