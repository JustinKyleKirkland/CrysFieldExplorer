# -*- coding: utf-8 -*-
"""
Test module for Crystal Field Explorer.

This module provides test functionality for comparing calculated results with experimental data,
including energy levels, intensities, magnetization, and susceptibility measurements.

Created on Sun Nov 19 21:28:23 2023
@author: qmc
"""

import math
from typing import List, Tuple


class Test:
    """
    Test class for comparing calculated crystal field results with experimental data.

    Attributes:
        exp_e: Experimental energy levels
        exp_I: Experimental intensities
        magnetization: Experimental magnetization data
        susceptibility: Experimental susceptibility data
    """

    def __init__(self, exp_e: List[float], exp_I: List[float], magnetization: List[float], susceptibility: List[float]):
        """
        Initialize the Test class with experimental data.

        Args:
            exp_e: List of experimental energy levels
            exp_I: List of experimental intensities
            magnetization: List of experimental magnetization values
            susceptibility: List of experimental susceptibility values
        """
        self.exp_e = exp_e
        self.exp_I = exp_I
        self.magnetization = magnetization
        self.susceptibility = susceptibility

    def energy_level_test(self, ev: List[float], rtol: float = 1e-9) -> Tuple[bool, List[bool]]:
        """
        Test calculated energy levels against experimental values.

        Args:
            ev: List of calculated energy levels
            rtol: Relative tolerance for comparison (default: 1e-9)

        Returns:
            Tuple containing:
                - Overall test result (bool)
                - List of individual level test results (List[bool])
        """
        if len(ev) != len(self.exp_e):
            raise ValueError("Length mismatch: Experimental and calculated energy levels must have the same length")

        results = []
        for i, (calc, exp) in enumerate(zip(ev, self.exp_e)):
            matches = math.isclose(calc, exp, rel_tol=rtol)
            results.append(matches)
            print(f"Energy level {i}: {'matches' if matches else 'does not match'} with experiment")

        return all(results), results

    def intensity_test(self, intensities: List[float], rtol: float = 1e-9) -> Tuple[bool, List[bool]]:
        """
        Test calculated intensities against experimental values.

        Args:
            intensities: List of calculated intensities
            rtol: Relative tolerance for comparison (default: 1e-9)

        Returns:
            Tuple containing:
                - Overall test result (bool)
                - List of individual intensity test results (List[bool])
        """
        if len(intensities) != len(self.exp_I):
            raise ValueError("Length mismatch: Experimental and calculated intensities must have the same length")

        results = []
        for i, (calc, exp) in enumerate(zip(intensities, self.exp_I)):
            matches = math.isclose(calc, exp, rel_tol=rtol)
            results.append(matches)
            print(f"Intensity level {i}: {'matches' if matches else 'does not match'} with experiment")

        return all(results), results
