"""Implementation of Coupled Map Lattices."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .constants import (
    DEFAULT_COUPLING,
    DEFAULT_N_MAPS,
    DEFAULT_TIME_STEPS,
    DEFAULT_WARMUP_STEPS,
)
from .maps import Map


@dataclass
class CMLConfig:
    """Configuration for Coupled Map Lattice."""

    map_function: Map
    n_maps: int = DEFAULT_N_MAPS
    coupling_strength: float = DEFAULT_COUPLING
    time_steps: int = DEFAULT_TIME_STEPS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    seed: int | None = None
    output_dir: str | None = None

    @property
    def map_name(self) -> str:
        """Get name of the map function."""
        return self.map_function.__class__.__name__

    def __post_init__(self):
        """Post-initialization checks."""
        if not (0 <= self.coupling_strength <= 1):
            raise ValueError("Coupling strength must be between 0 and 1.")


class CoupledMapLattice:
    """Implementation of a Coupled Map Lattice system."""

    def __init__(self, config: CMLConfig):
        """Initialize CML system.

        Args:
            config: Configuration object

        """
        self.config = config
        self._rng = np.random.default_rng(seed=config.seed)
        self.time_series = None

    def generate(self) -> npt.NDArray[np.float64]:
        """Generate time series from the CML.

        Returns:
            Array of shape (time_steps, n_maps) containing the time series

        """
        if not 0 <= self.config.coupling_strength <= 1:
            raise ValueError("Coupling strength must be between 0 and 1.")

        # Initialize with random values
        current_state = self._rng.random(size=self.config.n_maps)

        # Warm up the system
        for _ in range(self.config.warmup_steps):
            current_state = self._step(current_state)

        # Generate time series
        time_series = np.zeros((self.config.time_steps, self.config.n_maps))
        for t in range(self.config.time_steps):
            time_series[t] = current_state
            current_state = self._step(current_state)

        self.time_series = time_series
        return time_series

    def _step(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Perform one step of the CML evolution.

        Args:
            state: Current state array

        Returns:
            Next state

        """
        # Calculate the coupled term
        coupled_term = np.zeros_like(state)
        for i in range(len(state)):
            # Periodic boundary conditions
            left = state[i - 1] if i > 0 else state[-1]
            coupled_term[i] = (
                self.config.coupling_strength * left
                + (1 - self.config.coupling_strength) * state[i]
            )

        # Apply the map function
        return self.config.map_function(coupled_term)

    def save(self, filename: str | None = None) -> None:
        """Save time series to file.

        Args:
            filename: Output filename (optional)

        """
        if self.time_series is None:
            raise ValueError("No time series generated yet")

        if filename is None:
            if self.config.output_dir is None:
                raise ValueError("No output directory specified")
            filename = os.path.join(
                self.config.output_dir,
                f"""cml_{self.config.map_name}_\
                    {self.config.n_maps}_{self.config.coupling_strength}\
                    _{self.config.seed or 'noseed'}.npy""",
            )

        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        np.save(filename, self.time_series)

    @classmethod
    def load(cls, filename: str) -> npt.NDArray[np.float64]:
        """Load time series from file.

        Args:
            filename: Input filename

        Returns:
            Loaded time series

        """
        return np.load(filename)
