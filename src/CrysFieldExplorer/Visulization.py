# -*- coding: utf-8 -*-
"""Visualization module for CrysFieldExplorer.

This module provides visualization capabilities for various physical measurements
including susceptibility, magnetization, and neutron spectra.

Created on Sat Nov 18 13:06:05 2023
@author: qmc
"""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from CrysFieldExplorer.CrysFieldExplorer import Utilities

NumericArray = Union[List[float], np.ndarray]


class Visualizer:
    """Class for visualizing physical measurements data."""

    def __init__(self, font_size: int = 12, marker_size: int = 6):
        """Initialize the visualizer.

        Args:
            font_size: Size of font in plots (default: 12)
            marker_size: Size of markers in plots (default: 6)
        """
        self.font_size = font_size
        self.marker_size = marker_size

    def susceptibility(self, susceptibility_data: Tuple[NumericArray, NumericArray]) -> None:
        """Plot inverse susceptibility vs temperature.

        Args:
            susceptibility_data: Tuple of (temperature, susceptibility) arrays
        """
        temp, sus = susceptibility_data
        plt.figure()
        plt.plot(temp, 1 / sus, ".", markersize=self.marker_size)
        plt.xlabel("Temperature (K)", fontsize=self.font_size)
        plt.ylabel(r"$4\pi\chi\ (emu\ /cm^3 \times Oe^{-1})$", fontsize=self.font_size)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.title("Inverse Susceptibility")
        plt.show()

    def magnetization(self, magnetization_data: Tuple[NumericArray, NumericArray]) -> None:
        """Plot magnetization vs field.

        Args:
            magnetization_data: Tuple of (field, magnetization) arrays
        """
        field, mag = magnetization_data
        plt.figure()
        plt.plot(field, mag, ".", markersize=self.marker_size)
        plt.xlabel("Field (T)", fontsize=self.font_size)
        plt.ylabel("Magnetization (\mu_B)", fontsize=self.font_size)
        plt.xticks(fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.title("Magnetization")
        plt.show()

    def neutron_spectrum(self, energies: NumericArray, intensities: NumericArray, resolution: float) -> None:
        """Plot neutron spectrum with Lorentzian broadening.

        Args:
            energies: Array of energy values
            intensities: Array of intensity values
            resolution: Resolution/width parameter for Lorentzian broadening
        """
        if len(energies) != len(intensities):
            raise ValueError("Energies and intensities must have the same length")

        x_energies = np.linspace(0.8 * min(energies), 1.2 * max(energies), 100)
        total_intensity = np.zeros_like(x_energies)

        for energy, intensity in zip(energies, intensities):
            total_intensity += Utilities.lorentzian(x_energies, intensity, resolution, energy)

        plt.figure()
        plt.plot(x_energies, total_intensity)
        plt.xlabel("Energy (meV)", fontsize=self.font_size)
        plt.ylabel("Intensity (arb.unit)", fontsize=self.font_size)
        plt.title("Neutron Spectrum")
        plt.show()
