from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor


class Space(ABC):
    @abstractmethod
    def sample_positions(self, n: int) -> Tensor:
        """Sample positions from the space."""
        pass

    @abstractmethod
    def sample_velocities(self, positions: Tensor, p: float = 1) -> Tensor:
        """Sample velocities for a given set of positions."""
        pass

    @abstractmethod
    def clip(self, positions: Tensor) -> Tensor:
        """Clip positions within the bounds of the space."""
        pass

    @abstractmethod
    def __call__(self, points: Tensor) -> Tensor:
        """Evaluate the objective function at given points."""
        pass



class BoxSpace(Space):
    def __init__(self, mins: Tensor, maxs: Tensor, f: Callable[[Tensor], Tensor]) -> None:
        assert mins.dim() == maxs.dim() == 1, "mins and maxs must be 1D tensors"
        assert mins.size(0) == maxs.size(0), "mins and maxs must have the same shape"
        assert (mins < maxs).all(), "Each element in mins must be less than maxs"

        self.d = mins.size(0)
        self.mins = mins
        self.maxs = maxs
        self.dimensions = maxs - mins
        self.f = f

    def sample_positions(self, n: int) -> Tensor:
        """Samples `n` random positions within the box."""
        return torch.rand(n, self.d) * self.dimensions + self.mins

    def sample_velocities(self, positions: Tensor, p: float = 1) -> Tensor:
        """Samples `n` velocities, scaled by a factor `p`."""
        return (torch.rand_like(positions) - 0.5) * self.dimensions * 2 * p

    def clip(self, positions: Tensor) -> Tensor:
        """Clips positions to stay within bounds."""
        return torch.clamp(positions, min=self.mins, max=self.maxs)

    def __call__(self, points: Tensor) -> Tensor:
        """Evaluates function `f` at given points."""
        return self.f(points)        
