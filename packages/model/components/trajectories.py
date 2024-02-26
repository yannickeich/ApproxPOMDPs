import torch
from scipy.interpolate import interp1d
from packages.model.components.spaces import Space, DiscreteSpace, FiniteDiscreteSpace, ContinuousSpace
from abc import ABC, abstractmethod
from typing import Union


class Trajectory(ABC):
    def __init__(self, is_continuous_time: bool):
        """
        base class for a trajectory
        """
        self.is_continuous_time = is_continuous_time


class ContinuousTimeTrajectory(Trajectory):
    def __init__(self, times: torch.Tensor, values: torch.Tensor, interp_kind='previous'):
        self.times = times
        self.values = values
        super().__init__(is_continuous_time=True)
        self._interp_func = None
        self.interp_kind = interp_kind

    @property
    def interp_func(self):
        if self._interp_func is None:
            self._interp_func = interp1d(self.times, self.values, kind=self.interp_kind, axis=0,
                                         fill_value='extrapolate')
        return self._interp_func

    def __call__(self, time):
        return self.interp_func(time)


class DiscreteTimeTrajectory(Trajectory):
    def __init__(self, times: torch.Tensor, values: torch.Tensor):
        self.times = times
        self.values = values
        super().__init__(is_continuous_time=False)