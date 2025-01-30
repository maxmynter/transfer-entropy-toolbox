"""Expose core utilities."""
from .backend import Backend, get_backend, set_backend

MATRIX_DIMS = 2
VECTOR_DIMS = 1

__all__ = [

   "Backend",
   "get_backend",
   "set_backend"
]
