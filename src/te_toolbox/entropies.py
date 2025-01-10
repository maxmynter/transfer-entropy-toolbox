"""Contains all the entropy and derived measures."""

from functools import cache

import numpy as np
import numpy.typing as npt

MATRIX_DIMS = 2
VECTOR_DIMS = 1


@cache
def _discretize_data(
    data: tuple[float], bins: int | tuple[float, ...]
) -> tuple[npt.NDArray[np.int64], int]:
    """Convert dataset into discrete classes.

    Note: Values equal to outer bin edges handling:
    - Values equal to leftmost edge go into first bin
    - Values equal to rightmost edge go into last bin
    """
    data_array = np.array(data)
    if isinstance(bins, int):
        edges = np.linspace(data_array.min(), data_array.max(), bins + 1)
    else:
        edges = bins

    # Subtract 1 as digitize is 1 indexed
    indices = np.digitize(data_array, edges, right=False) - 1

    n_bins = len(edges) - 1

    # Make rightmost bin edge inclusive
    indices = np.where(indices == n_bins, n_bins - 1, indices)
    return indices, len(edges) - 1


def discrete_entropy(classes: npt.NDArray[np.int64], n_classes: int) -> np.float64:
    """Calculate the discrete entropy from class assignments."""
    probs = np.bincount(classes, minlength=n_classes) / len(classes)
    nonzero_mask = probs > 0
    return -np.sum(probs[nonzero_mask] * np.log(probs[nonzero_mask]))


def entropy(
    data: npt.NDArray[np.float64],
    bins: int | list[int | npt.NDArray[np.float64]] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | np.float64:
    """Calculate the entropy of one or more datasets.

    Args:
    ----
        data: Input data array. Can be 1D or 2D [timesteps x variables].
        bins: Number of bins for histogram or list of bin edges.

    Returns:
    -------
        float: Entropy value for 1D input
        ndarray: Array of entropy values for 2D input, one per variable

    Raises:
    ------
        ValueError: If data dimensions are invalid

    """
    if data.size == 0:
        raise ValueError("Cannot compute entropy of empty array")

    match data.ndim:
        case dim if dim == VECTOR_DIMS:
            hashable_bins = bins if isinstance(bins, int) else tuple(bins)
            indices, n_bins = _discretize_data(tuple(data), hashable_bins)
            return discrete_entropy(indices, n_bins)
        case dim if dim == MATRIX_DIMS:
            n_vars = data.shape[1]
            if isinstance(bins, int | float | np.ndarray):
                bins = [bins] * n_vars
            return np.array([entropy(data[:, i], bins[i]) for i in range(n_vars)])
        case _:
            raise ValueError(
                """
                Wrong data format.
                Data must be of dimension [timesteps] or [timesteps x variables]"""
            )


def discrete_joint_entropy(
    classes1: npt.NDArray[np.int64],
    classes2: npt.NDArray[np.int64],
    n_classes1: int,
    n_classes2: int,
) -> np.float64:
    """Calculate the discrete joint entropy from class assignments."""
    hist = np.zeros((n_classes1, n_classes2))

    np.add.at(hist, (classes1, classes2), 1)
    p_xy = hist / len(classes1)
    nonzero_mask = p_xy > 0
    return -np.sum(p_xy[nonzero_mask] * np.log(p_xy[nonzero_mask]))


def joint_entropy(
    data: np.ndarray,
    bins: int | list[int | npt.NDArray[np.float64]] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate pairwise joint entropy between all variables in the dataset.

    Args:
    ----
        data: Input data array of shape [timesteps x variables].
        bins: Number of bins or bin edges. Can be:
            - int: Same number of bins for all variables
            - [int, int]: Different number of bins for each variable
            - array: Bin edges for all variables
            - [array, array]: Different bin edges for each variable

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing joint entropies.
            Entry [i,j] is the joint entropy H(X_i, X_j).

    Raises:
    ------
        ValueError: If data has invalid dimensions or single variable.

    """
    if data.ndim != MATRIX_DIMS:
        raise ValueError("Data must be 2-dimensional [timesteps x variables]")

    dim = data.shape[1]  # Number of variables
    if dim < 2:  # noqa: PLR2004, 2 dimensions necessary for jent calculation
        raise ValueError("Need at least 2 variables to calculate joint entropy")

    jent = np.zeros((dim, dim), dtype=np.float64)

    if isinstance(bins, int | np.ndarray):
        bins = [bins] * dim

    idxs, jdxs = np.triu_indices(dim)
    for idx in range(len(idxs)):
        i, j = idxs[idx], jdxs[idx]
        hashable_bins_i = bins[i] if isinstance(bins[i], int) else tuple(bins[i])
        hashable_bins_j = bins[j] if isinstance(bins[j], int) else tuple(bins[j])

        indices_i, n_bins_i = _discretize_data(tuple(data[:, i]), hashable_bins_i)
        indices_j, n_bins_j = _discretize_data(tuple(data[:, j]), hashable_bins_j)

        entropy_value = discrete_joint_entropy(indices_i, indices_j, n_bins_i, n_bins_j)

        jent[i, j] = entropy_value
        jent[j, i] = entropy_value
    return jent


def discrete_multivar_joint_entropy(
    classes: list[npt.NDArray[np.int64]],
    n_classes: list[int],
) -> np.float64:
    """Calculate joint entropy from discrete classes for multiple variables."""
    hist = np.zeros(n_classes)

    idx = tuple(c for c in classes)
    np.add.at(hist, idx, 1)

    p = hist / len(classes[0])
    nonzero_mask = p > 0
    return -np.sum(p[nonzero_mask] * np.log(p[nonzero_mask]))


def multivar_joint_entropy(
    data: npt.NDArray[np.float64],
    bins: int | list[int | npt.NDArray[np.float64]] | npt.NDArray[np.float64],
) -> np.float64:
    """Calculate joint entropy for n variables.

    Args:
    ----
        data: Input array of shape [timesteps x variables].
        bins: Number of bins for all, for each, or list of bin edges for histogram.

    Returns:
    -------
        float: Joint entropy value H(X1,...,Xn).

    """
    classes = []
    n_classes = []
    for i in range(data.shape[1]):
        cur_bins = bins[i] if isinstance(bins, list) else bins
        indices, n_bins = _discretize_data(tuple(data[:, i]), cur_bins)
        classes.append(indices)
        n_classes.append(n_bins)

    return discrete_multivar_joint_entropy(classes, n_classes)


def conditional_entropy(
    data: np.ndarray,
    bins: int | list[int | npt.NDArray[np.float64]] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate conditional entropy between all pairs of variables.

    Uses the chain rule: H(Y|X) = H(X,Y) - H(X)

    Args:
    ----
        data: Input data array or DataFrame of shape [timesteps x variables].
        bins: Number of bins or bin edges (same formats as joint_entropy).

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing
            conditional entropies. Entry [i,j] is the conditional entropy H(X_i|X_j).

    Raises:
    ------
        ValueError: If data has invalid dimensions or single variable.

    """
    if data.ndim < MATRIX_DIMS:
        raise ValueError(
            "Need more than 2 time series to calculate conditional entropy"
        )
    h_xy = joint_entropy(data, bins)
    h_x = entropy(data, bins)

    return h_xy - h_x.reshape(1, -1)


def mutual_information(
    data: np.ndarray,
    bins: int | list[int | npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    norm: bool = True,
) -> npt.NDArray[np.float64]:
    """Calculate mutual information of time series.

    Args:
    ----
        data: Input data array of shape [timesteps x variables].
        bins: Number of bins or bin edges for histogram approximation of PDF.
        norm: Whether to normalize result between 0 and 1 using I(X,Y)/sqrt(H(X)*H(Y)).

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing
            mutual information values. Entry [i,j] is I(X_i,X_j).

    Raises:
    ------
        ValueError: If data has invalid dimensions or single variable.

    """
    if data.ndim == 1:
        raise ValueError("Cannot compute mutual information with single variable.")

    dim = data.shape[1]
    h_xy = joint_entropy(data, bins)
    h_x = entropy(data, bins)

    i, j = np.meshgrid(range(dim), range(dim))

    mi = h_x[i] + h_x[j] - h_xy[i, j]
    if norm:
        mi = np.divide(mi, np.sqrt(np.multiply(h_x[i], h_x[j])))

    return mi


def prepare_te_data(
    data: npt.NDArray[np.float64],
    lag: int,
    bins: int | list[int | npt.NDArray[np.float64]] | npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    list[int | npt.NDArray[np.float64]],
]:
    """Prepare data arrays and bins for transfer entropy calculation.

    Args:
    ----
        data: Input data array of shape [timesteps x variables].
        lag: Time lag for analysis.
        bins: Number of bins (int), bin edges, or list of per variable bin
              edges for histogram.

    Returns:
    -------
        tuple: (current data, lagged data, bin list)

    Raises:
    ------
        ValueError: If data dimensions are invalid.

    """
    if data.ndim != MATRIX_DIMS:
        raise ValueError("Data must be 2-dimensional [timesteps x variables]")
    if isinstance(bins, list) and len(bins) != data.shape[1]:
        raise ValueError(
            f"Bin specifications ({len(bins)}) must match variables ({data.shape[1]}). "
            "Provide either: a single integer for uniform bins, "
            "a list of integers for variable-specific bin counts, "
            "or a list of arrays defining bin edges for each variable."
        )

    dim = data.shape[1]
    bins = [bins] * dim if isinstance(bins, int) else bins
    return data[lag:], data[:-lag], bins


def transfer_entropy(
    data: npt.NDArray[np.float64],
    bins: int | list[int | npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    lag: int,
) -> npt.NDArray[np.float64]:
    """Calculate transfer entropy between all pairs of variables.

    Args:
    ----
        data: Input array of shape [timesteps x variables].
        bins: Number of bins or bin edges for histogram.
        lag: Time lag for analysis.

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing transfer
            entropy values. Entry [i,j] is the transfer entropy from X_j to X_i.

    """
    current, lagged, bins = prepare_te_data(data, lag, bins)
    dim = data.shape[1]
    tent = np.zeros((dim, dim))

    # H(X_t-1, Y_t-1)
    h_xy_lag = joint_entropy(lagged, bins)

    # H(X_t-1)
    h_x_lag = entropy(lagged, bins)

    for i in range(dim):
        # H(Y_t, Y_t-1)
        h_y_ylag = joint_entropy(
            np.column_stack([current[:, i], lagged[:, i]]),
            ([bins[i], bins[i]] if isinstance(bins, list) else bins),
        )[0, 1]

        for j in range(dim):
            # H(Y_t, Y_t-1, X_t-1)
            h_y_ylag_xlag = multivar_joint_entropy(
                np.column_stack([current[:, i], lagged[:, i], lagged[:, j]]),
                ([bins[i], bins[i], bins[j]] if isinstance(bins, list) else bins),
            )
            tent[i, j] = h_y_ylag + h_xy_lag[i, j] - h_y_ylag_xlag - h_x_lag[i]

    return tent


def normalized_transfer_entropy(
    data: npt.NDArray[np.float64],
    bins: int | list[int | npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    lag: int,
) -> npt.NDArray[np.float64]:
    """Calculate normalized transfer entropy between variables.

    Normalized as: 1 - H(Y_t | Y_t_lag, X_t_lag) / H(Y_t | Y_t_lag)
    where H(Y_t | Y_t_lag, X_t_lag) = H(Y_t, Y_t_lag, X_t_lag) - H(Y_t_lag, X_t_lag)
    and H(Y_t | Y_t_lag) = H(Y_t, Y_t_lag) - H(Y_t_lag).

    Args:
    ----
        data: Input array of shape [timesteps x variables].
        bins: Number of bins or bin edges for histogram.
        lag: Time lag for analysis.

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing normalized
            transfer entropy values. Entry [i,j] is between 0 and 1.

    Raises:
    ------
        ValueError: If data dimensions are invalid.

    """
    current, lagged, bins = prepare_te_data(data, lag, bins)
    dim = data.shape[1]
    tent = np.zeros((dim, dim))

    h_ylag_xlag = joint_entropy(lagged, bins)
    h_ylag = entropy(lagged, bins)

    for i in range(dim):
        h_y_ylag = multivar_joint_entropy(
            np.column_stack([current[:, i], lagged[:, i]]),
            ([bins[i], bins[i]] if isinstance(bins, list) else bins),
        )
        for j in range(dim):
            h_y_ylag_xlag = multivar_joint_entropy(
                np.column_stack([current[:, i], lagged[:, i], lagged[:, j]]),
                ([bins[i], bins[i], bins[j]] if isinstance(bins, list) else bins),
            )
            numerator = h_y_ylag_xlag - h_ylag_xlag[i, j]
            denominator = h_y_ylag - h_ylag[i]
            tent[i, j] = (
                np.round(1 - numerator / denominator, 10) if denominator != 0 else 0
            )
    return tent


def logn_normalized_transfer_entropy(
    data: npt.NDArray[np.float64],
    bins: int | list[int | npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    lag: int,
) -> npt.NDArray[np.float64]:
    """Calculate transfer entropy normalized by log(N) where N is number of bins.

    Args:
    ----
        data: Input array of shape [timesteps x variables].
        bins: Number of bins or bin edges for histogram.
        lag: Time lag for analysis.

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing
            logN-normalized transfer entropy values.

    """
    te = transfer_entropy(data, bins, lag)
    if isinstance(bins, list):
        for i in range(te.shape[0]):
            if isinstance(bins[i], int):
                te[i] = te[i] / np.log(bins[i])
            elif isinstance(bins[i], np.ndarray):
                n_bins = len(bins[i]) - 1
                te[i] = te[i] / np.log(n_bins)
            else:
                raise ValueError(
                    "Bin list must contain int of bins or np.array of bin edges"
                )
    elif isinstance(bins, np.ndarray):
        te /= np.log(len(bins) - 1)
    elif isinstance(bins, int):
        te /= np.log(bins)
    else:
        raise ValueError("Bins must be int, list[np.array], or np.array.")

    return te
