from abc import ABC, abstractmethod

from packages.model.components.spaces import Space, FiniteDiscreteSpace
from packages.model.components.transition_model import TabularTransitionModel, TransitionModel
from packages.model.components.observation_model import DiscreteTimeObservationModel, ObservationModel
import torch
from packages.model.components.filter.filter import ContDiscHistogramFilter
from packages.model.components.filter.discrete_time_obs.parametric_filter import Filter
import numpy as np

class Agent(ABC):
    def __init__(self, o_space: Space, a_space: Space):
        """
        Base class for an agent

        :param o_space: observation space
        :param a_space: action space
        """
        self.o_space = o_space
        self.a_space = a_space

        # Tensor for saving actions used in simulation
        self.action_vector = torch.Tensor()
        self.action_time_vector = torch.Tensor()

    @abstractmethod
    def get_action(self, time: torch.Tensor) -> torch.Tensor:
        # return action at a time point
        raise NotImplementedError

    @abstractmethod
    def add_observation(self, observation: torch.Tensor, time: torch.Tensor):
        # update the policy according to an observation
        raise NotImplementedError


class ConstantAgent(Agent):
    a_space: FiniteDiscreteSpace

    def __init__(self, o_space: Space, a_space: FiniteDiscreteSpace, action: torch.Tensor):
        """
        Implements a policy that always chooses the given action

        :param o_space: observation space
        :param a_space: action space
        :param action: action to choose for all times
        """
        Agent.__init__(self, o_space=o_space, a_space=a_space)
        self.action = action

    def get_action(self, time: torch.Tensor) -> torch.Tensor:
        return self.action

    def add_observation(self, observation: torch.Tensor, time: torch.Tensor):
        # ignore updates -> constant
        pass


class RandomAgent(Agent):
    a_space: FiniteDiscreteSpace

    # Implements a random action selection policy

    def get_action(self, time: torch.Tensor, num_samples: int = 1):
        actions = self.a_space.elements
        idx = torch.randint(actions.shape[0], size=(num_samples,))
        return actions[idx, :].squeeze(dim=0)

    def add_observation(self, observation, time):
        # ignore updates --> random
        pass


class BeliefBasedAgent(Filter, Agent, ABC):
    def __init__(self, t_model: TransitionModel, o_model: ObservationModel, initial_param: torch.Tensor,
                 initial_time: torch.Tensor, sim_method=None, sim_options=None, action_trajectory = None):
        """
        Base class for a agent based on a filter

        :param t_model: transition model
        :param o_model: observation model
        """
        Filter.__init__(self, t_model=t_model, o_model=o_model,
                        initial_param=initial_param, initial_time=initial_time, sim_method=sim_method,
                        sim_options=sim_options)
        Agent.__init__(self, o_space=o_model.o_space, a_space=t_model.a_space)
        self.action_trajectory = action_trajectory

    def get_action(self, time: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        # First get the belief at the current time, then get the corresponding action
        belief = self.get_belief(time)
        return self.belief_policy(belief, num_samples=num_samples,t=time)

    @abstractmethod
    def belief_policy(self, belief: torch.Tensor, num_samples: int = 1,t=None) -> torch.Tensor:
        # Maps the belief to an action
        raise NotImplementedError


class HistogramAgent(ContDiscHistogramFilter, BeliefBasedAgent):
    a_space: FiniteDiscreteSpace

    def __init__(self, t_model: TabularTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_hist: torch.Tensor,
                 initial_time: torch.Tensor, sim_method=None, sim_options=None,action_trajectory = None,Q_matrix = None, advantage_net = None):
        """
        Policy based on exact continuous discrete Histogram Filter

        :param t_model: tabular transition model
        :param o_model: discrete time observation model
        :param initial_hist: initial histogram
        :param initial_time: initial time point
        """
        ContDiscHistogramFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_hist,
                                         initial_time=initial_time, sim_method=sim_method, sim_options=sim_options)
        BeliefBasedAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_hist,
                                  initial_time=initial_time, sim_method=sim_method, sim_options=sim_options,action_trajectory = action_trajectory)

        self.Q_matrix = Q_matrix
        self.advantage_net = advantage_net



    def add_observation(self, observation: torch.Tensor, time: torch.Tensor):
        # Get the action at the current time point
        action = self.get_action(time)

        # Reset the filter belief
        self.reset(time, observation, action)

    def forward(self, time: torch.Tensor, histogram: torch.Tensor) -> torch.Tensor:
        # Get the action at the current time point
        action = self.belief_policy(histogram,t = time)

        self.action_time_vector = torch.cat((self.action_time_vector, time[None, ...].clone()))
        self.action_vector = torch.cat((self.action_vector, action[None, ...].clone()))

        # compute rhs of the master equation
        return self.drift(histogram, action).squeeze()

    def belief_policy(self, histogram: torch.Tensor, num_samples: int = 1,t=None) -> torch.Tensor:
        # Stub for the mapping



        if self.action_trajectory:
            return torch.tensor([self.action_trajectory(t)[0]]).long()

        if self.advantage_net != None:

            advantages = self.advantage_net(histogram)
            best = advantages.argmax()
            return self.a_space.elements[best]


        if self.Q_matrix != None:
            Q = (histogram * self.Q_matrix).sum(-1)
            best = Q.argmax()
            return self.a_space.elements[best]

        else:
            # Stub for the mapping
            if self.a_space.is_finite:
                actions = self.a_space.elements
                return actions[torch.tensor([1]), :].squeeze(dim=0)
            else:
                return torch.tensor([0.0])

