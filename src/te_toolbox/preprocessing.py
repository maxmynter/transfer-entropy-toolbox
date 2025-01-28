"""Preprocessing utilities for time series data."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def ft_surrogatization(
    data: npt.NDArray, rng: np.random.Generator | None = None
) -> npt.NDArray[np.float64]:
    """Generate Fourier transform surrogates of time series data.

    Creates phase-randomized surrogates that preserve the power spectrum
    (linear correlations) of the original data.

    Args:
        data: Input array of shape (n_samples, n_variables)
        rng: Random number generator instance

    Returns:
        Surrogate data array of same shape as input

    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples, n_vars = data.shape
    surrogates = np.empty_like(data)

    for i in range(n_vars):
        fft = np.fft.rfft(data[:, i])
        phases = 2 * np.pi * rng.random(len(fft))

        # Keep DC (0 freq) and Nyquist frequency phases unchanged
        phases[0] = 0.0
        if n_samples % 2 == 0:
            phases[-1] = 0.0

        fft_surr = fft * np.exp(1j * phases)
        surr = np.real(np.fft.irfft(fft_surr, n=n_samples))

        surrogates[:, i] = surr.ravel()

    return surrogates


def noisify(
    data: npt.NDArray[np.float64],
    noise_distribution: str | Callable,
    rng: np.random.Generator | None = None,
    amplitude: float = 0.5,
    **sampler_params,
) -> npt.NDArray[np.float64]:
    """Add noise following a distribution to data.

    Args:
    ----
        data: Input array of size (n_samples, n_variables)
        noise_distribution: Callable noise sampler or string name of distribution
        amplitude: Noise amplitude relative to data mean
        rng: Random number generatior
        **sampler_params: Additional parameters for noise sampler

    """
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(noise_distribution, str):
        builtin_samplers = {
            "normal": lambda size, **kwargs: rng.normal(size=size, **kwargs),
            "gaussian": lambda size, **kwargs: rng.normal(size=size, **kwargs),
            "uniform": lambda size, **kwargs: rng.uniform(size=size, **kwargs),
            "poisson": lambda size, **kwargs: rng.poisson(size=size, **kwargs),
        }

        if noise_distribution.lower() in builtin_samplers:
            noise_sampler = builtin_samplers[noise_distribution.lower()]
        else:
            raise ValueError("Unkown noise distribution")

    elif isinstance(noise_distribution, Callable):
        noise_sampler = noise_distribution
    else:
        raise ValueError(
            f"noise_sampler must be string or callable, got {noise_sampler}"
        )
    noisified = data.copy()
    n_samples, n_vars = data.shape

    means = np.mean(data, axis=0)
    noise = (
        amplitude
        * means[None, :]
        * noise_sampler((n_samples, n_vars), **sampler_params)
    )
    noisified += noise
    return noisified


def remap_to(
    data: npt.NDArray[np.float64],
    distribution: npt.NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """
    Perform a rank-ordered remapping of  distribution onto data.

    Args:
    ----
        data: Source data array of shape (n_timesteps, n_variables)
        distribution: Target distribution array of shape (n_timesteps, n_variables)
        rng: Optional random number generator for tie-breaking

    Returns:
    -------
        Remapped array with distribution's values following data's temporal pattern

    """
    if data.shape != distribution.shape:
        raise ValueError(
            f"Shape mismatch: data {data.shape} != distribution {distribution.shape}"
        )
    if rng is not None:
        # Add tiny random noise for tie-breaking
        random_noise = rng.random(size=data.shape) * 1e-10
        data_with_noise = data + random_noise
    else:
        data_with_noise = data

    ranks = np.argsort(np.argsort(data_with_noise, axis=0), axis=0)
    sorted_dist = np.sort(distribution, axis=0)

    return sorted_dist[ranks, np.arange(data.shape[1])]
