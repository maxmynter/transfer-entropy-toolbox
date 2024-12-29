import numpy as np
import numpy.typing as npt


def entropy(
    data: npt.NDArray[np.float64],
    bins: int | list[int] | npt.NDArray[np.float64],
):
    """Calculate the entropy of one or more datasets

    Args:
        data: Input data array. Can be 1D or 2D [timesteps x variables].
        bins: Number of bins for histogram or list of bin edges.

    Returns:
        float: Entropy value for 1D input
        ndarray: Array of entropy values for 2D input, one per variable

    Raises:
        ValueError: If data dimensions are invalid

    """
    if data.size == 0:
        raise ValueError("Cannot compute entropy of empty array")

    match data.ndim:
        case 1:
            hist, _ = np.histogram(data, bins = bins) 
            hist = hist / len(data)
            hist = np.where(hist==0, 1, hist)
            return -np.sum(hist * np.log(hist))
        case 2:
            n_vars = data.shape[1]
            if isinstance(bins, (int, float)):
                bins = [bins]* n_vars
            return np.array([entropy(data[:, i], bins[i]) for i in range(n_vars)])
        case _:
            raise ValueError("Wrong data format. Data must be of dimension [timesteps] or [timesteps x variables]")
