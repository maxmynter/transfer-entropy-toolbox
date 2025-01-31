"""Utilities for entropy calculations."""

from collections.abc import Callable
from typing import TypeVar

from .core.backend import Backend, get_backend

T = TypeVar("T")


def branch_funcs_by_backends(
    funcs: dict[Backend, Callable[..., T]], *args, **kwargs
) -> T:
    """Branch functions depending on backend."""
    if (backend := get_backend()) in funcs:
        return funcs[backend](*args, **kwargs)
    else:
        raise ValueError(f"Missing function for backend: {backend}")
