"""Tests for Coupled Map Lattice implementation."""

from pathlib import Path

import numpy as np
import pytest

from te_toolbox.systems.lattice import CMLConfig, CoupledMapLattice
from te_toolbox.systems.maps import TentMap


@pytest.fixture
def basic_config():
    """Create a basic CML configuration."""
    return CMLConfig(
        map_function=TentMap(r=2),
        n_maps=3,
        coupling_strength=0.1,
        time_steps=10,
        warmup_steps=5,
        seed=42,
    )


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary directory for output files."""
    output_dir = tmp_path / "cml_output"
    output_dir.mkdir()
    return str(output_dir)


def test_cml_initialization(basic_config):
    """Test CML initialization."""
    cml = CoupledMapLattice(basic_config)
    assert cml.config == basic_config
    assert cml.time_series is None


def test_cml_generation(basic_config):
    """Test time series generation."""
    cml = CoupledMapLattice(basic_config)
    time_series = cml.generate()

    # Check shape
    assert time_series.shape == (basic_config.time_steps, basic_config.n_maps)

    # Check values are in [0,1]
    assert np.all(time_series >= 0) and np.all(time_series <= 1)

    # Check time_series is stored
    assert np.array_equal(cml.time_series, time_series)


def test_cml_reproducibility(basic_config):
    """Test that same seed produces same results."""
    cml1 = CoupledMapLattice(basic_config)
    series1 = cml1.generate()

    cml2 = CoupledMapLattice(basic_config)
    series2 = cml2.generate()

    assert np.array_equal(series1, series2)


def test_cml_coupling(basic_config):
    """Test that coupling affects the system."""
    # No coupling
    config_uncoupled = CMLConfig(
        map_function=basic_config.map_function,
        n_maps=basic_config.n_maps,
        coupling_strength=0.0,
        time_steps=basic_config.time_steps,
        warmup_steps=basic_config.warmup_steps,
        seed=basic_config.seed,
    )

    # Full coupling
    config_coupled = CMLConfig(
        map_function=basic_config.map_function,
        n_maps=basic_config.n_maps,
        coupling_strength=1.0,
        time_steps=basic_config.time_steps,
        warmup_steps=basic_config.warmup_steps,
        seed=basic_config.seed,
    )

    cml_uncoupled = CoupledMapLattice(config_uncoupled)
    cml_coupled = CoupledMapLattice(config_coupled)

    series_uncoupled = cml_uncoupled.generate()
    series_coupled = cml_coupled.generate()

    # Results should be different with different coupling
    assert not np.array_equal(series_uncoupled, series_coupled)


def test_save_load(basic_config, tmp_output_dir):
    """Test saving and loading time series."""
    basic_config.output_dir = tmp_output_dir
    cml = CoupledMapLattice(basic_config)
    series = cml.generate()

    # Test save
    cml.save()

    # Check file exists
    files = list(Path(tmp_output_dir).glob("*.npy"))
    assert len(files) == 1

    # Test load
    loaded_series = CoupledMapLattice.load(str(files[0]))
    assert np.array_equal(series, loaded_series)


def test_error_handling(basic_config):
    """Test error handling."""
    cml = CoupledMapLattice(basic_config)

    # Test save before generate
    with pytest.raises(ValueError, match="No time series generated yet"):
        cml.save()

    # Test save without output_dir
    basic_config.output_dir = None
    cml.time_series = np.zeros((10, 3))  # Mock generated data
    with pytest.raises(ValueError, match="No output directory specified"):
        cml.save()


@pytest.mark.parametrize("coupling", [-0.1, 1.1])
def test_invalid_coupling(basic_config, coupling):
    """Test invalid coupling values."""
    basic_config.coupling_strength = coupling
    cml = CoupledMapLattice(basic_config)
    with pytest.raises(ValueError, match="Coupling strength must be between 0 and 1"):
        cml.generate()
