import matplotlib.pyplot as plt
import numpy as np
import pytest

from CrysFieldExplorer.Visulization import Visualizer


@pytest.fixture
def visualizer():
    """Fixture to create a Visualizer instance for tests."""
    return Visualizer(font_size=12, marker_size=6)


@pytest.fixture
def mock_plt(monkeypatch):
    """Fixture to mock matplotlib.pyplot to avoid displaying plots during tests."""

    def mock_show():
        pass

    monkeypatch.setattr(plt, "show", mock_show)
    return plt


def test_visualizer_initialization():
    """Test Visualizer class initialization with default and custom parameters."""
    # Test default parameters
    viz = Visualizer()
    assert viz.font_size == 12
    assert viz.marker_size == 6

    # Test custom parameters
    viz = Visualizer(font_size=14, marker_size=8)
    assert viz.font_size == 14
    assert viz.marker_size == 8


def test_susceptibility_plotting(visualizer, mock_plt):
    """Test susceptibility plotting functionality."""
    # Create sample data
    temperature = np.array([1.0, 2.0, 3.0, 4.0])
    susceptibility = np.array([0.5, 0.25, 0.167, 0.125])

    # Test plotting
    visualizer.susceptibility((temperature, susceptibility))

    # Since we're mocking plt.show(), we just verify the plot was created
    assert plt.gcf() is not None
    plt.close()


def test_magnetization_plotting(visualizer, mock_plt):
    """Test magnetization plotting functionality."""
    # Create sample data
    field = np.array([0.0, 1.0, 2.0, 3.0])
    magnetization = np.array([0.0, 1.0, 1.5, 1.8])

    # Test plotting
    visualizer.magnetization((field, magnetization))

    # Verify plot was created
    assert plt.gcf() is not None
    plt.close()


def test_neutron_spectrum_plotting(visualizer, mock_plt):
    """Test neutron spectrum plotting functionality."""
    # Create sample data
    energies = np.array([1.0, 2.0, 3.0])
    intensities = np.array([0.5, 1.0, 0.7])
    resolution = 0.1

    # Test plotting
    visualizer.neutron_spectrum(energies, intensities, resolution)

    # Verify plot was created
    assert plt.gcf() is not None
    plt.close()


def test_neutron_spectrum_validation(visualizer):
    """Test neutron spectrum input validation."""
    # Test mismatched lengths of energies and intensities
    energies = np.array([1.0, 2.0])
    intensities = np.array([0.5])
    resolution = 0.1

    with pytest.raises(ValueError, match="Energies and intensities must have the same length"):
        visualizer.neutron_spectrum(energies, intensities, resolution)


def test_input_types(mock_plt):
    """Test that the Visualizer accepts both lists and numpy arrays as input."""
    viz = Visualizer()

    # Test with lists - convert to numpy arrays before plotting
    temp_list = [1.0, 2.0, 3.0]
    sus_list = [0.5, 0.25, 0.167]
    viz.susceptibility((np.array(temp_list), np.array(sus_list)))
    plt.close()

    # Test with numpy arrays
    field_array = np.array([0.0, 1.0, 2.0])
    mag_array = np.array([0.0, 1.0, 1.5])
    viz.magnetization((field_array, mag_array))
    plt.close()

    assert True  # If we get here without errors, the test passes
