"""Tests for Coupled Map Lattice implementation."""

from pathlib import Path

import numpy as np
import pytest

from te_toolbox.systems.lattice import (
    CMLConfig,
    CoupledMapLattice,
    CoupledMapLatticeGenerator,
)
from te_toolbox.systems.maps import TentMap


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary directory for output files."""
    output_dir = tmp_path / "cml_output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def basic_config(tmp_output_dir):
    """Create a basic CML configuration."""
    return CMLConfig(
        map_function=TentMap(r=2),
        n_maps=3,
        coupling_strength=0.1,
        n_steps=10,
        warmup_steps=5,
        seed=42,
        output_dir=tmp_output_dir,
    )


@pytest.fixture
def basic_lattice(basic_config):
    """Create a basic CML for testing."""
    cml_generator = CoupledMapLatticeGenerator(basic_config)
    return cml_generator.generate()


def test_generator_initialization(basic_config):
    """Test generator initialization."""
    generator = CoupledMapLatticeGenerator(basic_config)
    assert generator.config == basic_config


def test_generator(basic_config):
    """Test the CML generator."""
    generator = CoupledMapLatticeGenerator(basic_config)
    cml = generator.generate()

    assert cml.lattice.shape == (
        basic_config.n_steps,
        basic_config.n_maps,
    ), "Specified and generated shapes don't match"


def test_generator_reproducibility(basic_config):
    """Test that same seed produces same results."""
    gen1 = CoupledMapLatticeGenerator(basic_config)
    lattice1 = gen1.generate()

    gen2 = CoupledMapLatticeGenerator(basic_config)
    lattice2 = gen2.generate()

    assert np.array_equal(lattice1.lattice, lattice2.lattice)


def test_generator_coupling(basic_config):
    """Test that coupling affects the system."""
    # No coupling
    config_uncoupled = CMLConfig(
        map_function=basic_config.map_function,
        n_maps=basic_config.n_maps,
        coupling_strength=0.0,
        n_steps=basic_config.n_steps,
        warmup_steps=basic_config.warmup_steps,
        seed=basic_config.seed,
    )

    # Full coupling
    config_coupled = CMLConfig(
        map_function=basic_config.map_function,
        n_maps=basic_config.n_maps,
        coupling_strength=1.0,
        n_steps=basic_config.n_steps,
        warmup_steps=basic_config.warmup_steps,
        seed=basic_config.seed,
    )

    gen_uncoupled = CoupledMapLatticeGenerator(config_uncoupled)
    gen_coupled = CoupledMapLatticeGenerator(config_coupled)

    lattice_uncoupled = gen_uncoupled.generate()
    lattice_coupled = gen_coupled.generate()

    assert not np.array_equal(lattice_uncoupled.lattice, lattice_coupled.lattice)


@pytest.mark.parametrize("coupling", [-0.1, 1.1])
def test_invalid_coupling(basic_config, coupling):
    """Test invalid coupling values."""
    basic_config.coupling_strength = coupling
    generator = CoupledMapLatticeGenerator(basic_config)
    with pytest.raises(ValueError, match="Coupling strength must be between 0 and 1"):
        generator.generate()


def test_save_load(basic_lattice, tmp_output_dir):
    """Test saving and loading time series."""
    basic_lattice.save()

    # Check file exists
    files = list(Path(tmp_output_dir).glob("cml_map=*"))
    assert len(files) == 1

    # Test load
    loaded_lattice = CoupledMapLattice.load(files[0])
    assert np.array_equal(basic_lattice.lattice, loaded_lattice.lattice)
    assert basic_lattice.map_name == loaded_lattice.map_name
    assert basic_lattice.coupling_strength == loaded_lattice.coupling_strength


def test_error_handling():
    """Test error handling."""
    # Test save without output_dir
    lattice = CoupledMapLattice(
        lattice=np.zeros((10, 3)),
        map_name="TentMap(r=2)",
        n_maps=3,
        n_steps=10,  # Add n_steps parameter
        coupling_strength=0.1,
        seed=42,
    )
    with pytest.raises(ValueError, match="No output directory specified"):
        lattice.save()

    # Test wrong number of timesteps
    with pytest.raises(ValueError, match="not match n_steps"):
        CoupledMapLattice(
            lattice=np.zeros((8, 3)),  # Wrong number of timesteps
            map_name="TentMap(r=2)",
            n_maps=3,
            n_steps=10,
            coupling_strength=0.1,
            seed=42,
        )

    # Test wrong number of maps
    with pytest.raises(ValueError, match="not match n_maps"):
        CoupledMapLattice(
            lattice=np.zeros((10, 4)),  # Wrong number of maps
            map_name="TentMap(r=2)",
            n_maps=3,
            n_steps=10,
            coupling_strength=0.1,
            seed=42,
        )


def test_path_handling(basic_lattice):
    """Test path handling in save method."""
    # Test with string path
    basic_lattice.save("test.npz")
    assert Path("test.npz").exists()

    # Test with Path object
    basic_lattice.save(Path("test.npz"))
    assert Path("test.npz").exists()

    # Cleanup
    Path("test.npz").unlink()


def test_metadata_preservation(basic_lattice, tmp_output_dir):
    """Test that metadata is preserved through save/load."""
    save_path = Path(tmp_output_dir) / "test.npz"
    basic_lattice.save(save_path)
    loaded = CoupledMapLattice.load(save_path)

    assert loaded.map_name == basic_lattice.map_name
    assert loaded.n_maps == basic_lattice.n_maps
    assert loaded.coupling_strength == basic_lattice.coupling_strength
    assert loaded.seed == basic_lattice.seed
