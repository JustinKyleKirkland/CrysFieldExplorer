import numpy as np
import pytest

from CrysFieldExplorer.Optimization import (
    CrystalFieldOptimizer,
    run_cma_optimization,
    setup_optimization_bounds,
)


@pytest.fixture
def sample_optimizer():
    """Create a sample CrystalFieldOptimizer instance for testing."""
    stevens_idx = [[2, 0], [2, 1], [2, 2]]
    alpha = 0.01
    beta = 0.02
    gamma = 0.003
    parameters = [1.0, 2.0, 3.0]
    temperature = 5.0
    field = [0.0, 0.0, 0.0]  # [Bx, By, Bz]
    true_eigenvalue = [1.0, 2.0, 3.0]
    true_intensity = [0.5, 0.3, 0.2]

    return CrystalFieldOptimizer(
        magnetic_ion="Er3+",
        stevens_idx=stevens_idx,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        parameters=parameters,
        temperature=temperature,
        field=field,
        true_eigenvalue=true_eigenvalue,
        true_intensity=true_intensity,
    )


def test_crystal_field_optimizer_initialization(sample_optimizer):
    """Test proper initialization of CrystalFieldOptimizer."""
    assert sample_optimizer.magnetic_ion == "Er3+"
    assert len(sample_optimizer.stevens_idx) == 3
    assert sample_optimizer.alpha == 0.01
    assert sample_optimizer.beta == 0.02
    assert sample_optimizer.gamma == 0.003
    assert len(sample_optimizer.parameters) == 3
    assert sample_optimizer.temperature == 5.0
    assert len(sample_optimizer.field) == 3  # [Bx, By, Bz]
    assert all(f == 0.0 for f in sample_optimizer.field)  # Zero field
    assert len(sample_optimizer.true_eigenvalue) == 3
    assert len(sample_optimizer.true_intensity) == 3


def test_calculate_eigenvalue_loss(sample_optimizer):
    """Test the eigenvalue loss calculation."""
    # Create sample Hamiltonian and eigenvalues
    dim = int(2 * sample_optimizer.J + 1)
    hamiltonian = np.eye(dim)
    eigenvalues = np.array([0.0, 1.0, 2.0])

    loss = sample_optimizer._calculate_eigenvalue_loss(hamiltonian, eigenvalues)
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_calculate_intensity_loss(sample_optimizer):
    """Test the intensity loss calculation."""
    # Create sample intensity values
    intensity = np.array([0.4, 0.3, 0.3])

    loss = sample_optimizer._calculate_intensity_loss(intensity)
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert loss >= 0.0


def test_cma_loss_single(sample_optimizer):
    """Test the single CMA loss calculation."""
    loss = sample_optimizer.cma_loss_single()
    assert isinstance(loss, float)
    assert not np.isnan(loss)


def test_cma_loss_single_fast(sample_optimizer):
    """Test the fast version of CMA loss calculation."""
    loss = sample_optimizer.cma_loss_single_fast()
    assert isinstance(loss, float)
    assert not np.isnan(loss)


def test_setup_optimization_bounds():
    """Test the setup of optimization bounds."""
    n_parameters = 5
    scale = 100.0
    lower_bound, upper_bound = setup_optimization_bounds(n_parameters, scale)

    assert len(lower_bound) == n_parameters
    assert len(upper_bound) == n_parameters
    assert np.all(lower_bound == -scale)
    assert np.all(upper_bound == scale)


def test_run_cma_optimization(sample_optimizer):
    """Test the CMA optimization function."""
    x_init = np.array([1.0, 2.0, 3.0])
    bounds = setup_optimization_bounds(len(x_init))

    # Create a simple test function that always returns a constant value
    def test_loss(*args):
        return 1.0

    # Replace the optimizer's loss function with our test function
    sample_optimizer.cma_loss_single_fast = test_loss

    # Run with very limited evaluations for testing
    opt_params, opt_loss = run_cma_optimization(
        optimizer=sample_optimizer,
        x_init=x_init,
        bounds=bounds,
        sigma=1e-7,
        max_fevals=10,  # Limited for testing
    )

    assert len(opt_params) == len(x_init)
    assert isinstance(opt_loss, float)
    assert not np.isnan(opt_loss)


def test_parameter_validation():
    """Test parameter validation during initialization."""
    with pytest.raises(ValueError, match="Number of parameters must match number of Stevens indices"):
        CrystalFieldOptimizer(
            magnetic_ion="Er3+",
            stevens_idx=[[2, 0], [2, 1]],  # 2 indices
            alpha=0.01,
            beta=0.02,
            gamma=0.003,
            parameters=[1.0, 2.0, 3.0],  # 3 parameters (mismatch)
            temperature=5.0,
            field=0.0,
            true_eigenvalue=[1.0, 2.0],
            true_intensity=[0.5, 0.5],
        )


def test_array_conversion():
    """Test proper conversion of input arrays to numpy arrays."""
    optimizer = CrystalFieldOptimizer(
        magnetic_ion="Er3+",
        stevens_idx=[[2, 0]],
        alpha=0.01,
        beta=0.02,
        gamma=0.003,
        parameters=[1.0],
        temperature=5.0,
        field=0.0,
        true_eigenvalue=[1.0],
        true_intensity=[1.0],
    )

    assert isinstance(optimizer.true_eigenvalue, np.ndarray)
    assert isinstance(optimizer.true_intensity, np.ndarray)


def test_field_initialization():
    """Test different field initialization formats."""
    # Test with single value (z-direction)
    optimizer = CrystalFieldOptimizer(
        magnetic_ion="Er3+",
        stevens_idx=[[2, 0]],
        alpha=0.01,
        beta=0.02,
        gamma=0.003,
        parameters=[1.0],
        temperature=5.0,
        field=1.0,  # Single value
        true_eigenvalue=[1.0],
        true_intensity=[1.0],
    )
    assert len(optimizer.field) == 3
    assert optimizer.field == [0.0, 0.0, 1.0]

    # Test with full vector
    optimizer = CrystalFieldOptimizer(
        magnetic_ion="Er3+",
        stevens_idx=[[2, 0]],
        alpha=0.01,
        beta=0.02,
        gamma=0.003,
        parameters=[1.0],
        temperature=5.0,
        field=[1.0, 2.0, 3.0],  # Vector
        true_eigenvalue=[1.0],
        true_intensity=[1.0],
    )
    assert len(optimizer.field) == 3
    assert optimizer.field == [1.0, 2.0, 3.0]
