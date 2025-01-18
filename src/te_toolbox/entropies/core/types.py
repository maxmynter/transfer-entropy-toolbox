"""Entropy type utilities."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BinType = int | FloatArray | Sequence[int | FloatArray]
NClasses = int | list[int]
