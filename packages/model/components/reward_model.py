from abc import ABC, abstractmethod
from packages.model.components.spaces import DiscreteSpace, Space
from packages.model.components.filter.discrete_time_obs.discrete_spaces.simple_discrete_families import PoissonFilter, BinomialFilter
import torch


class RewardModel(ABC):
    @abstractmethod
    def __call__(self, state: torch.Tensor, action: torch.Tensor,**kwargs):
        # return reward
        raise NotImplementedError

    def expected_reward(self, filter, belief, action: torch.Tensor,method='MC', **kwargs):
        def func(state):
            return self(state, action[...,None,:])
        ## Returns expected reward, depending on Projection Filter
        return filter.expectation_state_func(func,belief, method=method,**kwargs)


class QuadraticRewardModel(RewardModel):
    def __init__(self, goal,scale = 1):
        """
        Quadratic Reward Model
        Has functions to compute the reward, given the state of the system
        and the expected reward given the belief.

        :param goal: goal state for the system
        :param scale: Scale for the reward model
        """
        self.goal = goal
        self.scale = scale
    def __call__(self,state,action,**kwargs):
        """

        :param state:
        :param action:
        :param kwargs:
        :return:
        """
        return - (((state - self.goal)/self.scale) ** 2).mean(-1)

    def expected_reward(self,filter, belief, action: torch.Tensor, **kwargs):
        """
        This computes the expected reward.
        Closed Form solutions for binomial and poisson filter.


        :param filter:
        :param belief: binomial success probability parameters of the belief distribution
        :param action:
        :return: expected reward
        """
        if isinstance(filter, BinomialFilter):
            expected_reward = -(belief * filter.total_counts * (1 - belief) + (
                        filter.total_counts * belief) ** 2 - 2 * filter.total_counts * belief * self.goal + self.goal ** 2).sum(-1)
            return expected_reward/self.scale**2

        elif isinstance(filter, PoissonFilter):
            expected_reward = -((belief - self.goal) ** 2 + belief).sum(-1)
            return expected_reward/self.scale**2

        else:
            return RewardModel.expected_reward(self,filter,belief,action,**kwargs)
