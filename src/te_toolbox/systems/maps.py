"""Implementations of different chaotic maps."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Map(ABC):
    """Abstract base class for chaotic maps."""

    @abstractmethod
    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply the map function.

        Args:
            x: Input array

        Returns:
            Map result

        """
        pass

    @abstractmethod
    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the derivative of the map at point(s) x.

        Args:
            x: Input array

        Returns:
            Derivative of the map

        """
        pass


class TentMap(Map):
    """Tent map: f(x) = 2x if x < 0.5 else 2(1-x)."""

    def __init__(self, r: float):
        """Initialize tent map.

        Args:
            r: Growth parameter (default: 2.0
               consisten with doi.org/10.1103/PhysRevLett.85.461)

        """
        self.r = r

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply tent map.

        Args:
            x: Input array

        Returns:
            Tent map result

        """
        return np.where(x < 0.5, self.r * x, self.r * (1 - x))  # noqa: PLR2004 # Implementation of tent map

    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the derivative of the Tent map."""
        return np.where(x < 0.5, self.r, -self.r)  # noqa: PLR2004 # Implementation of tent map


class LogisticMap(Map):
    """Logistic map: f(x) = rx(1-x)."""

    def __init__(self, r: float = 4.0):
        """Initialize logistic map.

        Args:
            r: Growth parameter (default: 4.0 for chaotic regime)

        """
        self.r = r

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply logistic map.

        Args:
            x: Input array

        Returns:
            Logistic map result

        """
        return self.r * x * (1 - x)

    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the derivative of the Logistic map."""
        return self.r * (1 - 2 * x)


class BellowsMap(Map):
    """Implementation of the bellows map: f(x) = rx/(1-x)^b."""

    def __init__(self, r: float = 5.0, b: float = 6.0):
        """Initialize bellows map.

        Args:
            r: Map parameter (default: 5.0)
            b: Map parameter (default: 6.0) as in arxiv.org/pdf/2309.08449

        """
        self.r = r
        self.b = b

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply bellows map.

        Args:
            x: Input array

        Returns:
            Bellows map result

        """
        return (self.r * x) / (1 - x**self.b)

    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the derivative of the Bellows map."""
        return self.r * ((self.b - 1) * x**self.b + 1) / (1 - x**self.b) ** 2


class ExponentialMap(Map):
    """Implementation of the exponential map: f(x) = (exp(rx) - 1)/(exp(r) - 1)."""

    def __init__(self, r: float = 4.0):
        """Initialize exponential map.

        Args:
            r: Growth parameter (default: 4.0)

        """
        self.r = r

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply exponential map.

        Args:
            x: Input array

        Returns:
            Exponential map result

        """
        return (np.exp(self.r * x) - 1) / (np.exp(self.r) - 1)

    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the derivative of the Exponential map."""
        return np.exp(self.r * (1 - x)) * (1 - self.r * x)
