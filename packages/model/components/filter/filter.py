from abc import ABC, abstractmethod
from packages.model.components.transition_model import TransitionModel, TabularTransitionModel
from packages.model.components.observation_model import ObservationModel, DiscreteTimeObservationModel
import torch
from torchdiffeq import odeint
import torch.nn as nn
import numpy as np
from typing import Callable


class Filter(nn.Module, ABC):
    def __init__(self, t_model: TransitionModel, o_model: ObservationModel, initial_param: torch.Tensor,
                 initial_time: torch.Tensor, sim_method = None, sim_options = None):
        """
        Base class for filters

        :param t_model: transition model
        :param o_model: observation model
        """
        self.param_dim = initial_param.shape[-1]
        self.t_model = t_model
        self.o_model = o_model
        self.belief_vector = initial_param.clone()[None, ...]
        self.time_vector = torch.Tensor([initial_time])

        if sim_method is None:
            sim_method ='rk4'
        self.sim_method = sim_method
        if sim_options is None:
            sim_options = dict(step_size=0.001)
        self.sim_options = sim_options

        nn.Module.__init__(self)

    def belief_to_unc_belief(self, belief):
        """
        Transforms the parameter from a constraint bounded space to the real numbers
        """
        # Default Identity Mapping
        return belief

    def unc_belief_to_belief(self, unc_belief):
        """
        Transforms the unconstraint parameter to the constraint bounded parameter space
        """

        # Default Identity Mapping
        return unc_belief

    def get_belief(self, t):
        """
        Gives belief of parameters for input time.
        Belief is the solution to an ODE for the parameters

        :param t: current time
        :return: belief at the time point
        """
        if torch.isclose(self.time_vector[-1], t):
            return self.belief_vector[-1]

        if t< self.time_vector[-1]:
            # TODO Implement
            # Interpolate
            raise NotImplementedError
            # return ...
        # integrate using unconstraint space
        times = torch.arange(self.time_vector[-1],t,self.sim_options['step_size'])
        if times[-1]!=t:
            times = torch.cat((times,t[None]))
        sol = odeint(func=self, y0=self.belief_to_unc_belief(self.belief_vector[-1]),
                     t=times, method=self.sim_method, options=self.sim_options)

        self.time_vector = torch.cat((self.time_vector, times[1:].clone()))
        self.belief_vector = torch.cat((self.belief_vector, self.unc_belief_to_belief(sol[1:].clone())))
        return self.belief_vector[-1]

    @abstractmethod
    def expectation_state_func(self, func: Callable[[torch.Tensor], torch.Tensor], belief: torch.Tensor, **kwargs):
        # TODO make abstract and implement in sub-methods
        raise NotImplementedError

    @abstractmethod
    def drift(self, belief, action, **kwargs):
        # TODO make abstract and implement in sub-methods
        raise NotImplementedError

    @abstractmethod
    def jump_update(self, belief, observation, action):
        # TODO make abstract and implement in sub-methods
        raise NotImplementedError

    @abstractmethod
    def forward(self, t, unc_belief):
        # integrate using unconstraint space
        raise NotImplementedError

    def reset(self, time, observation, action):
        """
        Reset the filter at the current time point

        :param time: time point of the observation
        :param observation: observation signal
        :param action: action signal
        """
        if time < self.time_vector[-1]:
            # TODO Implement
            # Delete all beliefs self.t_hist_vector>time
            raise NotImplementedError
        # Update belief until given time point
        belief = self.get_belief(time)

        # Update belief with new observation
        belief_new = self.jump_update(belief, observation, action)

        self.time_vector = torch.cat((self.time_vector, time[None, ...].clone()))
        self.belief_vector = torch.cat((self.belief_vector, belief_new[None, ...].clone()))


class ContDiscHistogramFilter(Filter, ABC):
    t_model: TabularTransitionModel
    o_model: DiscreteTimeObservationModel

    # Exact continuous discrete histogram filter

    def jump_update(self, hist, observation, action,**kwargs):
        """
        Computes new belief after an observation for a given action

        :param hist: [belief_batch x belief_dim]
        :param observation: [action_batch x obs_batch * belief_batch  x obs_dim]
        :param action: [action_batch x obs_batch * belief_batch  x action_dim]
        :return: [belief_batch x belief_dim] belief after reset
        """

        state = self.t_model.s_space.elements  # dim belief_dim x action_batch x obs_batch * belief_batch  x belief_dim x state_dim

        # Compute posterior
        log_like = self.o_model.log_prob(state, action[..., None, :], observation[..., None,
                                                                      :])  # dim belief_dim x action_batch x obs_batch * belief_batch
        hist_new = (log_like + hist.log()).exp()

        # normalize
        hist_new /= hist_new.sum(-1, keepdims=True)

        return hist_new

    def drift(self, hist, action, **kwargs):
        """
        Maps to a function that returns the current drift.
        :param hist: belief at the current time point
        :param action: action of the current time point
        :return: drift

        Master equation: Rhs of ODE for prior dynamic
        :param hist: histogram
        :param action: action signal
        """
        if action.ndim > 1:
            action_idx = np.ravel_multi_index(action.flatten(end_dim=-2).numpy().T,
                                              self.t_model.a_space.cardinalities).tolist()
        else:
            action_idx = [np.ravel_multi_index(action.numpy().T, self.t_model.a_space.cardinalities)]

        table = self.t_model.table[..., action_idx].permute(2, 0, 1)
        table = table.reshape(*action.shape[:-1], table.shape[-2], table.shape[-1])
        # action_batch x belief_batch x state x next_state

        out = (table * hist[..., None]).sum(-2)

        return out  # action_batch x belief_batch x next_state

    def expectation_state_func(self, func: Callable[[torch.Tensor], torch.Tensor], belief: torch.Tensor, **kwargs):
        """
        Computes the expectation of the input function under the given belief histogram
        :param func: function, of which expectation is calculated
        ouput of function needs to be of the form ...x...x state_batch, so that @ with belief works

        :param belief: histogram
        :param args:
        :param kwargs: potentially arguments needed for the function like "action=.."
        :return: expectation
        """
        state = self.t_model.s_space.elements
        weights = belief
        return (func(state.expand(*weights.shape, state.shape[-1])) * weights).sum(-1)
