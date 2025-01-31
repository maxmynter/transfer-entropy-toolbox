"""Type stubs for fast entropy calculations implemented in C++."""

from collections.abc import Sequence

import numpy as np

from ..core.types import IntArray

Float = np.float64

def discrete_conditional_entropy(data: IntArray, n_classes: Sequence[int]) -> Float: ...
def discrete_entropy(data: IntArray, n_classes: int) -> Float: ...
def discrete_joint_entropy(data: IntArray, n_classes: Sequence[int]) -> Float: ...
def discrete_multivar_joint_entropy(
    classes: Sequence[IntArray], n_classes: Sequence[int]
) -> Float: ...
def discrete_transfer_entropy(
    data: IntArray, n_classes: Sequence[int], lag: int
) -> Float: ...
def discrete_log_normalized_transfer_entropy(
    data: IntArray, n_classes: Sequence[int], lag: int
) -> Float: ...
