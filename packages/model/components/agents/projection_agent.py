from abc import ABC

import numpy as np
import torch

from packages.model.components.agents.agent import BeliefBasedAgent
from packages.model.components.filter.cont_time_obs.cont_time_filter import ContTimeProjectionFilter
from packages.model.components.filter.discrete_time_obs.discrete_spaces.discrete_parametric_filter import DiscreteProjectionFilter, FiniteProjectionFilter
from packages.model.components.filter.discrete_time_obs.parametric_filter import DiscreteTimeProjectionFilter
from packages.model.components.observation_model import DiscreteTimeObservationModel, ExactContTimeObservationModel
from packages.model.components.spaces import FiniteDiscreteSpace
from packages.model.components.transition_model import TransitionModel, DiscreteTransitionModel, \
    ContinuousTransitionModel


class ProjectionAgent(DiscreteTimeProjectionFilter, BeliefBasedAgent, ABC):
    a_space: FiniteDiscreteSpace

    def __init__(self, t_model: TransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix=None,Q_function=None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None, jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent class for an Agent based on a Projection Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        DiscreteTimeProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                              initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                              device=device, sim_method=sim_method, sim_options=sim_options,
                                              jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options)
        BeliefBasedAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                  initial_time=initial_time, sim_method=sim_method, sim_options=sim_options,action_trajectory = action_trajectory)

        self.advantage_net = advantage_net
        self.Q_matrix = Q_matrix
        self.Q_function = Q_function
        self.normalization = normalization

    def add_observation(self, observation: torch.Tensor, time: torch.Tensor):
        # Get the action at the current time point
        action = self.get_action(time)
        # Reset the filter belief
        self.reset(time, observation, action)

    def belief_policy(self, belief: torch.Tensor, num_samples: int = 1,t=None):
        """
        Returns an action given the current belief.
        If agent posseses an action trajectory, that it is supposed to follow, it chooses it.
        If agent possesses an advantage mapping (learnt via collocation), it uses the mapping.
        If the agent has learnt Q-Function from the underlying MDP, it makes an estimate of the Q-values, given the belief and chooses the action to maximize it.
        If these two mappings do not exist, the agent chooses a constant action.

        :param belief:
        :param num_samples:
        :return:
        """
        if self.action_trajectory:
            return torch.tensor([self.action_trajectory(t)[0]]).long()

        if self.advantage_net != None:

            advantages = self.advantage_net(belief)
            best = advantages.argmax()
            return self.a_space.elements[best]

        if self.Q_function:
            def evaluate_Q(state):
                #Bring state dim to the back for expectation function
                return self.Q_function(state.float()/self.normalization).T
            Q = self.expectation_state_func(evaluate_Q,belief,method=self.exp_method,num_samples=self.exp_samples)
            value, action = Q.max(0)
            return torch.tensor([action])

        if self.Q_matrix != None:
            #states = self.sample(param=belief, num_samples=self.exp_samples)
            def get_Q_values(states):
                try:
                    index = np.ravel_multi_index(states.int().permute(-1, *range(len(states.shape[:-1]))).numpy(),
                                            self.t_model.s_space.cardinalities)
                except:
                    index = (states.reshape(-1, states.shape[-1])[..., None, :] == self.t_model.s_space.elements).prod(-1).nonzero()[
                        ..., -1]
                Q = self.Q_matrix[:, index]
                return Q
            if self.exp_method == 'exact':
                Q = (self.log_prob(self.t_model.s_space.elements, belief).exp() * self.Q_matrix).sum(-1)
            else:
                Q = self.expectation_state_func(get_Q_values, belief, method=self.exp_method, num_samples=self.exp_samples)
            best = Q.argmax()
            return self.a_space.elements[best]

        else:
            # Stub for the mapping
            if self.a_space.is_finite:
                actions = self.a_space.elements
                return actions[torch.tensor([1]), :].squeeze(dim=0)
            else:
                return torch.tensor([0.0])

    def forward(self, t, unc_param):
        # integrate using unconstraint space

        # Get the action at the current time point
        # param is already transformed
        param = self.unc_belief_to_belief(unc_param)
        action = self.belief_policy(param,t=t)
        self.action_time_vector = torch.cat((self.action_time_vector, t[None, ...].clone()))
        self.action_vector = torch.cat((self.action_vector, action[None, ...].clone()))
        #if t>1.35:
        #    print(param)
        # compute rhs of parameter ODE
        param_grad_state = param.requires_grad
        param.requires_grad_(True)
        with torch.enable_grad():
            out = self.belief_to_unc_belief(param)

            grad = torch.autograd.grad(outputs=out.sum(), inputs=param)[0]
        param.requires_grad_(param_grad_state)

        return grad * self.drift(param, action)


class ContTimeProjectionAgent(ContTimeProjectionFilter, BeliefBasedAgent, ABC):
    a_space: FiniteDiscreteSpace

    def __init__(self, t_model: TransitionModel,
                 o_model: ExactContTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix=None, Q_function = None,normalization=1,
                 exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None, jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent class for an Agent based on a Projection Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        ContTimeProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                  initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                  device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        BeliefBasedAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                  initial_time=initial_time, sim_method=sim_method, sim_options=sim_options,action_trajectory=action_trajectory)

        self.advantage_net = advantage_net
        self.Q_matrix = Q_matrix
        self.Q_function = Q_function
        self.normalization = normalization

    def add_observation(self, observation: torch.Tensor, time: torch.Tensor):
        # Get the action at the current time point
        action = self.get_action(time)
        # Reset the filter belief
        self.reset(time, observation, action)

    def belief_policy(self, belief: torch.Tensor, num_samples: int = 1,t=None):
        """
        Returns an action given the current belief.
        If agent possesses an advantage mapping (learnt via collocation), it uses the mapping.
        If the agent has learnt Q-Function from the underlying MDP, it makes an estimate of the Q-values, given the belief and chooses the action to maximize it.
        If these two mappings do not exist, the agent chooses a constant action.
        :param belief:
        :param num_samples:
        :return:
        """
        if self.action_trajectory:
            return torch.tensor([self.action_trajectory(t)[0]]).long()

        if self.advantage_net != None:
            advantages = self.advantage_net(belief)
            best = advantages.argmax()
            return self.a_space.elements[best]

        if self.Q_function:
            def evaluate_Q(state):
                #Bring state dim to the back for expectation function
                return self.Q_function(state.float()/self.normalization).T
            Q = self.expectation_state_func(evaluate_Q,belief,method=self.exp_method,num_samples=self.exp_samples)
            value, action = Q.max(0)
            return torch.tensor([action])

        if self.Q_matrix != None:
            #states = self.sample(param=belief, num_samples=self.exp_samples)
            def get_Q_values(states):
                try:
                    index = np.ravel_multi_index(states.int().permute(-1, *range(len(states.shape[:-1]))).numpy(),
                                             self.t_model.s_space.cardinalities)
                except:
                    index = (states.reshape(-1, states.shape[-1])[..., None, :] == self.t_model.s_space.elements).prod(-1).nonzero()[
                        ..., -1]

                Q = self.Q_matrix[:, index]
                return Q
            #Overwriting, because indexing is slow for multinomial case
            if self.exp_method == 'exact':
                Q = (self.log_prob(self.t_model.s_space.elements, belief).exp() * self.Q_matrix).sum(-1)
            else:
                Q = self.expectation_state_func(get_Q_values, belief, method=self.exp_method, num_samples=self.exp_samples)
            best = Q.argmax()
            return self.a_space.elements[best]
        else:
            # Stub for the mapping
            if self.a_space.is_finite:
                actions = self.a_space.elements
                return actions[torch.tensor([1]), :].squeeze(dim=0)
            else:
                return torch.tensor([0.0])

    def forward(self, t, unc_param):
        # integrate using unconstraint space

        # Get the action at the current time point
        # param is already transformed
        param = self.unc_belief_to_belief(unc_param)
        action = self.belief_policy(param,t=t)
        self.action_time_vector = torch.cat((self.action_time_vector, t[None, ...].clone()))
        self.action_vector = torch.cat((self.action_vector, action[None, ...].clone()))
        #if t>1.35:
        #    print(param)
        # compute rhs of parameter ODE
        param_grad_state = param.requires_grad
        param.requires_grad_(True)
        with torch.enable_grad():
            out = self.belief_to_unc_belief(param)

            grad = torch.autograd.grad(outputs=out.sum(), inputs=param)[0]
        param.requires_grad_(param_grad_state)

        return grad * self.drift(param, action)


class DiscreteProjectionAgent(DiscreteProjectionFilter, ProjectionAgent, ABC):
    t_model: DiscreteTransitionModel

    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None, normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        DiscreteProjectionFilter.__init__(self, t_model, o_model, initial_param, initial_time, exp_method=exp_method,
                                          exp_samples=exp_samples, device=device, sim_method=sim_method,
                                          sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        ProjectionAgent.__init__(self, t_model, o_model, initial_param, initial_time, advantage_net, Q_matrix,Q_function = Q_function,normalization=normalization,
                                 exp_method=exp_method, exp_samples=exp_samples, device=device, sim_method=sim_method,
                                 sim_options=sim_options,
                                 jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options,
                                 action_trajectory = action_trajectory)


class FiniteProjectionAgent(FiniteProjectionFilter, DiscreteProjectionAgent, ABC):
    t_model: DiscreteTransitionModel

    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None, normalization=1,exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        FiniteProjectionFilter.__init__(self, t_model, o_model, initial_param, initial_time, exp_method=exp_method,
                                        exp_samples=exp_samples, device=device, sim_method=sim_method,
                                        sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        DiscreteProjectionAgent.__init__(self, t_model, o_model, initial_param, initial_time, advantage_net, Q_matrix,Q_function = Q_function,normalization=normalization,
                                         exp_method=exp_method, exp_samples=exp_samples, device=device,
                                         sim_method=sim_method, sim_options=sim_options,
                                         jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options,
                                         action_trajectory=action_trajectory)


