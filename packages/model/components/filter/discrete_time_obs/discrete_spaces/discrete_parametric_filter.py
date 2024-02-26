from packages.model.components.transition_model import DiscreteTransitionModel
from packages.model.components.observation_model import ObservationModel, DiscreteTimeObservationModel
import torch
from abc import ABC, abstractmethod
from packages.model.components.filter.discrete_time_obs.parametric_filter import DiscreteTimeProjectionFilter



#Make sure that it is the same in all files
EPS = 1e-4


class DiscreteProjectionFilter(DiscreteTimeProjectionFilter, ABC):
    t_model: DiscreteTransitionModel
    o_model: DiscreteTimeObservationModel

    def __init__(self, t_model: DiscreteTransitionModel, o_model: ObservationModel, initial_param: torch.Tensor,
                 initial_time: torch.Tensor, exp_method='MC', exp_samples=1000, device=None, sim_method=None,
                 sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Base class for a projection type filter using a discrete parametrization
        :param t_model:
        :param o_model:
        :param initial_param:
        :param initial_time:
        :param device:
        """

        DiscreteTimeProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                              initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                              device=device, sim_method=sim_method, sim_options=sim_options,
                                              jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options)

    def operator_grad_log(self, state, action, param):
        """
         Computes the master equation adjoint operator applied to the grad log of the prob. Equation 11 in Leo Bronsteins paper on entropic matching.
         This is needed for the ODE that describes the parameters.
         The method needs the 'rates' method of the transition model to return rates and next_states.

         :param param: current belief of parameter
         :param state: state, here batch for all possible states is used
         :param action: action signal
         :return: dim action_batch x belief_batch x state_batch x state_dim
         """

        # rates Dimension num_evnts x belief_batch*state_batch x action_batch
        # next_states  Dimension num_events x belief_batch*state_batch x action_batch x state_dim
        rates, next_states = self.t_model.rates(state, action)
        change_vectors = self.grad_log_prob(next_states, param[None, ...]) - self.grad_log_prob(state, param)
        sum = (change_vectors * rates[..., None]).sum(dim=0)
        return sum


class FiniteProjectionFilter(DiscreteProjectionFilter, ABC):
    t_model: DiscreteTransitionModel
    o_model: DiscreteTimeObservationModel

    def __init__(self, t_model: DiscreteTransitionModel, o_model: ObservationModel, initial_param: torch.Tensor,
                 initial_time: torch.Tensor, exp_method='MC', exp_samples=1000, device=None, sim_method=None,
                 sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Base Class for a projection type filter using a finite discrete parametrization

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial parameter for simulation
        :param initial_time: initial time for simulation
        """

        DiscreteTimeProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                              initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                              device=device, sim_method=sim_method, sim_options=sim_options,
                                              jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options)

    @abstractmethod
    def support_bounds(self) -> [torch.Tensor, torch.Tensor]:
        """
        Returns the lower and upper bounds for the support of the filter distribution
        """
        raise NotImplementedError

    def expectation_state_func(self, function, param: torch.Tensor, method='MC', **kwargs):
        """
        Computes the expectation of the input function, under the belief given input param

        different methods: exact, unscented, unscented_round, unscented_heuristic
        This function adds exact, as it is not part of the ProjectionFilter function, as it only works for discrete spaces


        :param function:  function, of which expectation is calculated
                        ouput of function needs to be of the form ...x...x belief_batch x state_batch, because weight will be of form belief_batch x state_batch
        :param param: parameter of pmf
        :return: expectation
        """
        if method == 'exact':
            # log_prob belief_batch x state_batch
            state = self.t_model.s_space.elements.to(self.device)
            weights = self.log_prob(state, param[..., None, :]).exp()
            return (function(state.expand(*weights.shape, state.shape[-1])) * weights).sum(-1)

        else:
            return DiscreteTimeProjectionFilter.expectation_state_func(self, function, param, method=method, **kwargs)


