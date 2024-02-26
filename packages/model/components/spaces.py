from abc import ABC
from typing import Tuple
import torch


class Space(ABC):
    def __init__(self, is_finite: bool, dimensions: int):
        """
        Base class for a space
        :param is_finite: is the state space finite
        :param dimensions: number of dimensions
        """
        self.is_finite = is_finite
        self.dimensions = dimensions


class DiscreteSpace(Space):
    def __init__(self, dimensions: int, is_finite=False):
        """
        Base class for a countable space

        :param dimensions: number of dimensions
        """
        super().__init__(is_finite=is_finite, dimensions=dimensions)


class FiniteDiscreteSpace(DiscreteSpace):
    def __init__(self, cardinalities: Tuple[int, ...]):
        """
        Finite state space class

        :param cardinalities: cardinality for each dimension
        """
        super().__init__(dimensions=cardinalities.__len__(), is_finite=True)
        self.cardinalities = cardinalities
        self._elements = None

    @property
    def elements(self):  # enumeration for all elements in the finite set --> saves tensor in memory
        if self._elements is None:
            self._elements = torch.cartesian_prod(
                *[torch.arange(cardinality) for cardinality in self.cardinalities]).view(-1, self.dimensions)
        return self._elements

class MultinomialSpace(DiscreteSpace):
    def __init__(self,total_N,dimensions):
        """

        :param total_N:
        :param dimensions:
        """
        super().__init__(dimensions=dimensions,is_finite=True)
        self.total_N = total_N
        self._elements = None

    @property
    def elements(self):  # enumeration for all elements in the finite set --> saves tensor in memory

        ##TODO change from naive implementation to a combinatorial expression
        ## Naive implemenation saves cartesian product of dimensions, and then keeps only the ones that add up to N
        if self._elements is None:
            self._elements = torch.cartesian_prod(
                *[torch.arange(self.total_N+1) for i in range(self.dimensions)]).view(-1, self.dimensions)
            self._elements = self._elements[self._elements.sum(-1)==self.total_N]
        return self._elements


class ContinuousSpace(Space):
    def __init__(self, dimensions: int):
        """
        Base class for a countable space

        :param dimensions: number of dimensions
        """
        super().__init__(is_finite=False, dimensions=dimensions)