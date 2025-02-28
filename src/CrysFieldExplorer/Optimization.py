# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 02:50:17 2023

@author: qmc
"""

from typing import Dict, List, Tuple, Union

import cma
import numpy as np
from mpi4py import MPI

import CrysFieldExplorer.CrysFieldExplorer as crs

ArrayLike = Union[np.ndarray, List[float]]


class CrystalFieldOptimizer(crs.CrysFieldExplorer):
    """Crystal Field Optimization class using CMA-ES algorithm.

    This class extends CrysFieldExplorer to provide optimization capabilities
    for crystal field parameters using experimental data.
    """

    def __init__(
        self,
        magnetic_ion: str,
        stevens_idx: List[List[int]],
        alpha: float,
        beta: float,
        gamma: float,
        parameters: Union[Dict[str, float], ArrayLike],
        temperature: float,
        field: Union[float, List[float]],  # Can be single value or [Bx, By, Bz]
        true_eigenvalue: ArrayLike,
        true_intensity: ArrayLike,
    ) -> None:
        """Initialize the Crystal Field Optimizer.

        Args:
            magnetic_ion: Name of the magnetic ion
            stevens_idx: Stevens operator indices
            alpha: Stevens alpha parameter
            beta: Stevens beta parameter
            gamma: Stevens gamma parameter
            parameters: Crystal field parameters
            temperature: Temperature in Kelvin
            field: Applied magnetic field (single value for z-direction or [Bx, By, Bz])
            true_eigenvalue: Experimental energy levels
            true_intensity: Experimental intensities

        Raises:
            ValueError: If parameters length doesn't match stevens_idx length
        """
        # Convert single field value to [0, 0, Bz] format
        field_vector = [0.0, 0.0, field] if isinstance(field, (int, float)) else field

        # Validate parameters
        if isinstance(parameters, dict):
            if len(parameters) != len(stevens_idx):
                raise ValueError("Number of parameters must match number of Stevens indices")
        elif len(parameters) != len(stevens_idx):
            raise ValueError("Number of parameters must match number of Stevens indices")

        # Call parent class constructor first
        super().__init__(magnetic_ion, stevens_idx, alpha, beta, gamma, parameters, temperature, field_vector)

        # Store optimization-specific attributes
        self.true_eigenvalue = np.array(true_eigenvalue)
        self.true_intensity = np.array(true_intensity)

    def test(self):
        print(self.parameters)

    def _calculate_eigenvalue_loss(self, hamiltonian: np.ndarray, eigenvalues: np.ndarray) -> float:
        """Calculate the loss based on eigenvalue differences.

        Args:
            hamiltonian: The Hamiltonian matrix
            eigenvalues: Calculated eigenvalues

        Returns:
            float: Log of the absolute determinant difference
        """
        loss = 0.0
        eye_matrix = np.eye(int(2 * self.J + 1))

        for true_val in self.true_eigenvalue:
            det = np.linalg.det((true_val + eigenvalues[0]) * eye_matrix - hamiltonian)
            loss += det * det

        return np.log10(np.abs(loss))

    def _calculate_intensity_loss(self, intensity: np.ndarray) -> float:
        """Calculate the normalized intensity loss.

        Args:
            intensity: Calculated intensity values

        Returns:
            float: Normalized root mean square error
        """
        diff_squared = np.sum((self.true_intensity - intensity) ** 2)
        norm = np.sum(self.true_intensity**2)
        return np.sqrt(diff_squared / norm)

    def cma_loss_single(self, *args) -> float:
        """Calculate the loss function for CMA optimization.

        Returns:
            float: Total loss value
        """
        # Calculate Hamiltonian and eigenvalues
        eigenvalues, eigenvectors, hamiltonian = self.Hamiltonian()

        # Calculate losses
        ev_loss = self._calculate_eigenvalue_loss(hamiltonian, eigenvalues)

        return ev_loss

    def cma_loss_single_fast(self, *args) -> float:
        """Calculate a faster version of the loss function including intensity.

        Returns:
            float: Combined loss value from eigenvalues and intensities
        """
        # Calculate Hamiltonian and eigenvalues
        eigenvalues, eigenvectors, hamiltonian = self.Hamiltonian()

        # Calculate neutron scattering intensities more efficiently
        intensities = self.Neutron_Intensity_fast(1, len(self.true_intensity))

        # Calculate both losses
        ev_loss = self._calculate_eigenvalue_loss(hamiltonian, eigenvalues)
        int_loss = self._calculate_intensity_loss(intensities)

        return ev_loss + 10 * int_loss

    def cma_loss_single_fast_mag(self) -> float:
        """Calculate loss function optimized for magnetic calculations.

        Returns:
            float: Loss value based on magnetic properties
        """
        eigenvalues, eigenvectors, hamiltonian = self.magsolver()
        return self._calculate_eigenvalue_loss(hamiltonian, eigenvalues)

    def Neutron_Intensity_fast(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Calculate neutron scattering intensities more efficiently.

        This is a faster version of the neutron intensity calculation
        that only computes the required transitions.

        Args:
            start_idx: Starting transition index
            end_idx: Ending transition index

        Returns:
            numpy.ndarray: Array of calculated intensities
        """
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors, _ = self.Hamiltonian()

        # Calculate transition matrix elements
        intensities = []
        for i in range(start_idx, end_idx + 1):
            intensity = np.abs(np.dot(eigenvectors[:, 0], eigenvectors[:, i])) ** 2
            intensities.append(intensity)

        return np.array(intensities)


def setup_optimization_bounds(n_parameters: int, scale: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """Set up the optimization bounds for CMA-ES.

    Args:
        n_parameters: Number of parameters to optimize
        scale: Scale factor for the bounds

    Returns:
        Tuple containing lower and upper bounds
    """
    lower_bound = -np.ones(n_parameters) * scale
    upper_bound = np.ones(n_parameters) * scale
    return lower_bound, upper_bound


def run_cma_optimization(
    optimizer: CrystalFieldOptimizer,
    x_init: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    sigma: float = 1e-7,
    max_fevals: int = 10000000,
) -> Tuple[np.ndarray, float]:
    """Run CMA-ES optimization.

    Args:
        optimizer: Instance of CrystalFieldOptimizer
        x_init: Initial parameter values
        bounds: Tuple of (lower_bounds, upper_bounds)
        sigma: Initial step size
        max_fevals: Maximum number of function evaluations

    Returns:
        Tuple of (optimized_parameters, final_loss)
    """
    options = {
        "maxfevals": max_fevals,
        "tolfacupx": 1e9,
        "bounds": bounds,
    }

    result = cma.fmin(
        optimizer.cma_loss_single_fast,
        x_init,
        sigma,
        options=options,
        restarts=1,
        restart_from_best=True,
        incpopsize=1,
        eval_initial_x=True,
        bipop=True,
    )

    return result[0], result[1]


def run_parallel_optimization(
    magnetic_ion: str,
    stevens_idx: List[List[int]],
    alpha: float,
    beta: float,
    gamma: float,
    temperature: float,
    field: float,
    true_eigenvalue: ArrayLike,
    true_intensity: ArrayLike,
    n_tries: int = 1,
    output_file: str = "optimization_results.csv",
) -> None:
    """Run parallel optimization using MPI.

    Args:
        magnetic_ion: Name of the magnetic ion
        stevens_idx: Stevens operator indices
        alpha: Stevens alpha parameter
        beta: Stevens beta parameter
        gamma: Stevens gamma parameter
        temperature: Temperature in Kelvin
        field: Applied magnetic field
        true_eigenvalue: Experimental energy levels
        true_intensity: Experimental intensities
        n_tries: Number of optimization attempts
        output_file: Path to save results
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_parameters = len(stevens_idx)
    lower_bound, upper_bound = setup_optimization_bounds(n_parameters)

    results = np.zeros((n_tries, n_parameters + 2))

    for i in range(n_tries):
        # Generate random initial parameters
        x_init = np.random.uniform(low=lower_bound, high=upper_bound, size=n_parameters)

        # Create optimizer instance
        optimizer = CrystalFieldOptimizer(
            magnetic_ion=magnetic_ion,
            stevens_idx=stevens_idx,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            parameters=x_init,
            temperature=temperature,
            field=field,
            true_eigenvalue=true_eigenvalue,
            true_intensity=true_intensity,
        )

        # Run optimization
        opt_params, opt_loss = run_cma_optimization(
            optimizer=optimizer,
            x_init=x_init,
            bounds=(lower_bound, upper_bound),
        )

        results[i, :n_parameters] = opt_params
        results[i, n_parameters] = opt_loss
        results[i, n_parameters + 1] = rank

    # Gather results from all processes
    if rank == 0:
        all_results = results
        for source in range(1, size):
            received = comm.recv(source=source)
            all_results = np.vstack((all_results, received))
        np.savetxt(output_file, all_results, fmt="%2.20e", delimiter=",")
    else:
        comm.send(results, dest=0)

    comm.Barrier()


if __name__ == "__main__":
    # Example usage
    stevens_idx = [
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

    # Stevens parameters
    alpha = 0.01 * 10.0 * 4 / (45 * 35)
    beta = 0.01 * 100.0 * 2 / (11 * 15 * 273)
    gamma = 0.01 * 10.0 * 8 / (13**2 * 11**2 * 3**3 * 7)

    # Experimental data
    true_eigenvalue = np.array([1.77, 5.25, 7.17, 13.72, 22.58, 27.81, 49.24])
    true_intensity = np.array([1.00, 0.365, 0.000, 0.167, 0.074, 0.027, 0.010])

    run_parallel_optimization(
        magnetic_ion="Er3+",
        stevens_idx=stevens_idx,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        temperature=5,
        field=0,
        true_eigenvalue=true_eigenvalue,
        true_intensity=true_intensity,
        n_tries=1,
        output_file="Er_optimization_results.csv",
    )
