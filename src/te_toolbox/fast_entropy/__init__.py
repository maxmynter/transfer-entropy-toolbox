"""Package for fast entropy calculations implemented in C++."""

from ._fast_entropy import (
    discrete_entropy,
    discrete_joint_entropy,
    discrete_multivar_joint_entropy,
)

__all__ = [
    "discrete_entropy",
    "discrete_joint_entropy",
    "discrete_multivar_joint_entropy",
]
