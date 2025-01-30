"""Configure whether to use the CPP or Python backend for entropy calculation."""

from enum import Enum


class Backend(Enum):
    """Configuration of the entropy calculation backend."""

    CPP = "cpp"
    PY = "python"


_backend = Backend.CPP


def set_backend(backend: str) -> None:
    """Set the global backend variable."""
    global _backend  # noqa: PLW0603 # Use the global state to configure the backend
    try:
        _backend = next(b for b in Backend if b.value == backend.lower())
    except StopIteration:
        valid_backends = [b.value for b in Backend]
        raise ValueError(
            f"Unknown backend: {backend}. Valid backends are: {valid_backends}"
        ) from None


def get_backend() -> Backend:
    """Get the currently set backend variable."""
    return _backend
