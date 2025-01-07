"""Script to generate CML data for various maps and coupling strengths."""

from pathlib import Path

import numpy as np

from te_toolbox.systems.lattice import CMLConfig, CoupledMapLatticeGenerator
from te_toolbox.systems.maps import (
    BellowsMap,
    ExponentialMap,
    LogisticMap,
    Map,
    TentMap,
)

OUTPUT_DIR = Path("data/")
SEED = 42
N_TRANSIENT = 10**4
N_MAPS = 100
N_ITER = 300
LAG = 1
EPSILONS = np.linspace(0, 1, 20)
N_BINS = np.arange(2, 31)


def create_cml(map_function: Map) -> None:
    """Generate CML data for a given map across different coupling strengths.

    Args:
        map_function: Map to use for the CML

    """
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate data for each coupling strength
    for eps in EPSILONS:
        print(f"Generating {map_function.__class__.__name__} with eps={eps:.2f}")

        # Create configuration
        config = CMLConfig(
            map_function=map_function,
            n_maps=N_MAPS,
            coupling_strength=eps,
            n_steps=N_ITER,
            warmup_steps=N_TRANSIENT,
            seed=SEED,
            output_dir=str(OUTPUT_DIR),
        )

        # Generate and save data
        generator = CoupledMapLatticeGenerator(config)
        cml = generator.generate()
        cml.save()


def main():
    """Generate data for all maps."""
    maps = [LogisticMap(r=4), BellowsMap(r=5, b=6), ExponentialMap(r=4), TentMap(r=2)]

    for map_function in maps:
        create_cml(map_function)


if __name__ == "__main__":
    main()
