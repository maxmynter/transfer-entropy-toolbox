"""Analysis script for epsilon scan."""

from pathlib import Path

from eps_scan_constants import (
    EPS_DATA_DIR,
    EPSILONS,
    N_BINS,
    N_ITER,
    N_MAPS,
    PLOT_PATH,
    RELATIVE_NOISE_AMPLITUDE,
    maps,
)
from plot_utils import analyze_map, plot_results, save_results


def main():
    """Plot the Epsilon Scan results."""
    for map_name in maps:
        print(f"__________________{map_name}")
        prefix = (
            f"Noise_{RELATIVE_NOISE_AMPLITUDE}_GaussianRemap_"
            f"{N_ITER}Data_{len(EPSILONS)}eps_{map_name}_"
        )
        plot_prefix = str(PLOT_PATH) + "/" + prefix
        save_prefix = Path(EPS_DATA_DIR) / Path("te_by_eps/")
        results = analyze_map(
            map_name, plot_prefix, add_noise_flag=RELATIVE_NOISE_AMPLITUDE > 0
        )
        save_results(results, save_prefix / "te_by_eps.json")

        plot_results(results, plot_prefix, N_MAPS, N_ITER, EPSILONS, N_BINS)


if __name__ == "__main__":
    main()
