"""Implementation of Coupled Map Lattices."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .constants import (
    DEFAULT_COUPLING,
    DEFAULT_N_MAPS,
    DEFAULT_N_STEPS,
    DEFAULT_WARMUP_STEPS,
)
from .maps import Map


@dataclass
class CMLConfig:
    """Configuration for Coupled Map Lattice."""

    map_function: Map
    n_maps: int = DEFAULT_N_MAPS
    coupling_strength: float = DEFAULT_COUPLING
    n_steps: int = DEFAULT_N_STEPS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    seed: int | None = None
    output_dir: Path | None = None

    @property
    def map_name(self) -> str:
        """Get name of the map function."""
        return repr(self.map_function)

    def __post_init__(self):
        """Post-initialization checks and coercions."""
        if not (0 <= self.coupling_strength <= 1):
            raise ValueError("Coupling strength must be between 0 and 1.")
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)


@dataclass(frozen=True)
class CoupledMapLattice:
    """Data class storing the CML system."""

    lattice: npt.NDArray[np.float64]
    map_name: str
    n_maps: int
    n_steps: int
    coupling_strength: float
    seed: int | None
    generation_date: str = field(
        default_factory=lambda: np.datetime64("now").astype(str)
    )
    output_dir: Path | None = None

    def __post_init__(self):
        """Validate the lattice shape."""
        if self.lattice.shape[0] != self.n_steps:
            raise ValueError(
                f"Lattice steps {self.lattice.shape[0]} do "
                f"not match n_steps={self.n_steps}"
            )
        if self.lattice.shape[1] != self.n_maps:
            raise ValueError(
                f"Lattice shape {self.lattice.shape[1]} "
                f"does not match n_maps={self.n_maps}"
            )

    @classmethod
    def generate_default_filename(
        cls,
        map_name: str,
        n_maps: int,
        n_steps: int,
        coupling_strength: float,
        seed: int | None,
    ) -> str:
        """Generate a default filename for CML containing hyperparameters."""
        return (
            f"cml_map={map_name}_{n_maps}_x_{n_steps}_"
            f"eps={coupling_strength}_seed={seed or 'noseed'}.npz"
        )

    @property
    def default_filename(self) -> Path:
        """Generate default filename containing the generation parameters."""
        filename = CoupledMapLattice.generate_default_filename(
            self.map_name, self.n_maps, self.n_steps, self.coupling_strength, self.seed
        )
        if self.output_dir:
            return self.output_dir / filename
        else:
            return Path(filename)

    def save(self, filename: Path | str | None = None) -> None:
        """Save lattice and metadata.

        Args:
        ----
            filename: Output filename (optional, will use defaultif not provided)

        """
        if filename is None and self.output_dir is None:
            raise ValueError("No output directory specified")
        save_path = self.default_filename if filename is None else Path(filename)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "map_name": self.map_name,
            "n_maps": self.n_maps,
            "n_steps": self.n_steps,
            "coupling_strength": self.coupling_strength,
            "seed": -1 if self.seed is None else self.seed,
            "generation_date": self.generation_date,
        }

        np.savez_compressed(save_path, lattice=self.lattice, metadata=metadata)
        print(f"Saved {save_path}")

    @classmethod
    def load(cls, filename: Path | str) -> "CoupledMapLattice":
        """Load time series and metadata from npz file.

        Args:
        ----
            filename: Path to npz file

        Returns:
        -------
            CMLData object containing the loaded data

        """
        filename = Path(filename)
        with np.load(filename, allow_pickle=True) as data:
            lattice = data["lattice"]
            metadata = data["metadata"].item()

            return cls(
                lattice=lattice,
                map_name=metadata["map_name"],
                n_steps=metadata["n_steps"],
                n_maps=metadata["n_maps"],
                coupling_strength=metadata["coupling_strength"],
                seed=None if metadata["seed"] == -1 else metadata["seed"],
                generation_date=metadata["generation_date"],
            )


class CoupledMapLatticeGenerator:
    """Generator class for Coupled Map Lattice systems."""

    def __init__(self, config: CMLConfig):
        """Initialize CML system.

        Args:
        ----
            config: Configuration object

        """
        self.config = config
        self._rng = np.random.default_rng(seed=config.seed)

    def generate(self) -> CoupledMapLattice:
        """Generate time series from the CML.

        Returns
        -------
            CoupledMapLattice object containing the generated system and metadata

        """
        if not 0 <= self.config.coupling_strength <= 1:
            raise ValueError("Coupling strength must be between 0 and 1.")

        # Initialize with random values
        current_state = self._rng.random(size=self.config.n_maps)

        # Warm up the system
        for _ in range(self.config.warmup_steps):
            current_state = self._step(current_state)

        lattice = np.zeros((self.config.n_steps, self.config.n_maps))
        for t in range(self.config.n_steps):
            lattice[t] = current_state
            current_state = self._step(current_state)

        return CoupledMapLattice(
            lattice=lattice,
            n_steps=self.config.n_steps,
            map_name=self.config.map_name,
            n_maps=self.config.n_maps,
            coupling_strength=self.config.coupling_strength,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
        )

    def _step(self, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Perform one step of the CML evolution.

        Args:
        ----
            state: Current state array

        Returns:
        -------
            Next state

        """
        left_neighbors = np.roll(state, 1)
        coupled_term = (
            self.config.coupling_strength * left_neighbors
            + (1 - self.config.coupling_strength) * state
        )

        return self.config.map_function(coupled_term)
