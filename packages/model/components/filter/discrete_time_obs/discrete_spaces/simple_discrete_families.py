from abc import ABC

import torch

from packages.model.components.filter.discrete_time_obs.discrete_spaces.discrete_parametric_filter import FiniteProjectionFilter, EPS, \
    DiscreteProjectionFilter
from packages.model.components.observation_model import DiscreteTimeObservationModel
from packages.model.components.spaces import FiniteDiscreteSpace
from packages.model.components.transition_model import DiscreteTransitionModel


class BinomialFilter(FiniteProjectionFilter, ABC):
    t_model: DiscreteTransitionModel
    o_model: DiscreteTimeObservationModel

    def belief_to_unc_belief(self, param):
        return torch.logit(param)

    def unc_belief_to_belief(self, unc_param):
        return torch.sigmoid(unc_param).clip_(min=EPS, max=1 - EPS)

    def __init__(self, t_model: DiscreteTransitionModel, o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor, initial_time: torch.Tensor, exp_method='MC', exp_samples=1000,
                 device=None, sim_method=None, sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Class for binomial filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial success probabilties of binomials
        :param initial_time: initial time point
        """

        FiniteProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                        initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                        device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)

        assert isinstance(self.t_model.s_space, FiniteDiscreteSpace), "StateSpace should be Finite for BinomialFilter"
        self.total_counts = torch.tensor(self.t_model.s_space.cardinalities) - 1
        self.distribution = torch.distributions.Binomial(self.total_counts, initial_param)
    def inv_fisher_matrix(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the inverse fisher matrix given the parameters

        :param param: success probabilty of binomials
        :return: inverse fisher matrix. Dim: belief_batch x state_dim x state_dim
        """
        # if param.ndim == 1:
        #    param = param[None, :]
        vec = (param - param ** 2) / self.total_counts
        return torch.diag_embed(vec)

    def fisher_matrix(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
        # if param.ndim == 1:
        #    param = param[None, :]

        vec = (param - param ** 2) / self.total_counts
        return torch.diag_embed(1. / vec)

    def operator_grad_log(self, state, action, param):
        """
        Computes the master equation operator applied to the grad log of the prob.
        This is needed for the ODE that describes the parameters.
        The method needs the 'rates' method of the transition model to return rates and next_states.

        :param param: current belief of parameter
        :param state: state, here batch for all possible states is used
        :param action: action signal
        :return: dim action_batch x belief_batch x state_batch x state_dim
        """
        rates, next_states = self.t_model.rates(state, action)

        change_vectors = next_states - state
        # change_vectors Dimension num_events x action_batch x belief_batch x state_batch x state_dim
        sum = (change_vectors * rates[..., None]).sum(
            dim=0)  # torch.sum(change_vechange_vectorsctors * rates.permute(0, 2, 1)[..., None], (0))
        return sum / (param - param ** 2)

    def log_prob(self, state, param):
        """
        Returns the log of the pmf of the model, given parameter and state

        :param param: parameter (success probabilties)
        :return: log probabilties, dim belief_batch x state_batch
        """
        # self.distribution.probs = param
        # self.distribution.logits = param.logit()
        # q = self.distribution.log_prob(state)
        q = torch.distributions.Binomial(self.total_counts, param).log_prob(state)

        return q.sum(-1)

    def sample(self, param, num_samples=1):
        # self.distribution.probs = param
        # self.distribution.logits = param.logit()
        # self.distribution.total_count = self.distribution.total_count.expand(*param.shape)
        # state = self.distribution.sample((num_samples,))
        state = torch.distributions.Binomial(self.total_counts, param).sample((num_samples,))

        return state.permute(*range(1, len(param.shape[:-1]) + 1), 0, -1)

    def moments(self, param):
        mean = self.total_counts * param
        var = mean * (1.0 - param)
        skew = var * (1.0 - 2.0 * param)
        return mean, var, skew

    def support_bounds(self) -> [torch.Tensor, torch.Tensor]:
        return torch.zeros_like(self.total_counts), self.total_counts

    def jump_update(self, param, observation, action,**kwargs):

        try:
            method = kwargs['method']
        except KeyError:
            method = self.exp_method
        try:
            num_samples = kwargs['num_samples']
        except KeyError:
            num_samples = self.exp_samples


        def numfunc(state):
            return ((self.o_model.log_prob(state, action[..., None, :],
                                           observation[..., None, :])).exp()[None]*state.permute(-1,*range(0,len(state.shape[:-1]))))
        def denfunc(state):
            return (self.o_model.log_prob(state, action[..., None, :],observation[..., None, :])).exp()


        num=self.expectation_state_func(numfunc,param,method=method,num_samples=num_samples)
        den=self.expectation_state_func(denfunc,param,method=method,num_samples=num_samples)
        param_new = num / den
        param_new = param_new.permute(*range(1,len(param_new.shape)),0)
        param_new = 1 / self.total_counts * param_new

        ### get rid of nan. If there is one use old param for the update
        if param_new.isnan().sum() > 0:
            if param.dim() == 1:
                param_new[:] = param
            else:
                idx,_  =param_new.isnan().max(0)
                param_new[param_new.isnan()] = param[idx[None]]
        if param_new.isinf().sum() > 0:
            if param.dim() == 1:
                param_new[:] = param
            else:
                idx, _ = param_new.isinf().max(0)
                param_new[param_new.isinf()] = param[idx[None]]


        return param_new.clip_(min=EPS, max=1 - EPS)

    def drift(self, belief, action, **kwargs):
        """
        Overwriting ProjectionFilter.drift, as the inverse of the fisher matrix can be computed analytically.
        This function therefore calls the inverse directly instead of calling torch,solve(fisher_matrix...)
        Compute the right hand side of the ODE for the parameter.
        :param belief:
        :param action:
        :param kwargs:
        :return:
        """
        # If not called otherwise, use filters expectation method
        try:
            method = kwargs['method']
        except KeyError:
            method = self.exp_method
        try:
            num_samples = kwargs['num_samples']
        except KeyError:
            num_samples = self.exp_samples

        # dim before permute: action_batch x state_dim x belief_batch
        # output dim action_batch x belief_batch x state_dim
        def func2(state):
            return self.increment(state, action[..., None, :], belief[..., None, :], method=method).permute(-1, *range(
                len(state.shape[:-1])))

        def func(state):
            return self.operator_grad_log(state, action[..., None, :], belief[..., None, :]).permute(-1, *range(
                len(state.shape[:-1])))

        ## To compare. this one does the fisher matrix inside the expectation, if increment(including fisher_matrix) can be written in closed form, one should overwrite drift
        # drift = self.expectation_state_func(func2, belief, method=method).permute(*range(1, len(belief.shape[:-1]) + 1),0)

        drift = (self.inv_fisher_matrix(belief, method=method, num_samples=num_samples) @
                 self.expectation_state_func(func, belief, method=method, num_samples=num_samples).permute(
                     *range(1, len(belief.shape[:-1]) + 1),
                     0)[..., None]).sum(dim=-1)

        return drift


class PoissonFilter(DiscreteProjectionFilter):
    t_model: DiscreteTransitionModel
    o_model: DiscreteTimeObservationModel

    def belief_to_unc_belief(self, param):
        return torch.log(param)

    def unc_belief_to_belief(self, unc_param):
        return torch.exp(unc_param)

    def __init__(self, t_model: DiscreteTransitionModel, o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor, initial_time: torch.Tensor, exp_method='MC', exp_samples=1000,
                 device=None, sim_method=None, sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Class for poisson filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial rate parameters for poisson distribution
        :param initial_time: initial time point
        """

        DiscreteProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                          initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                          device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        self.distribution = torch.distributions.Poisson(initial_param)

    def inv_fisher_matrix(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the inverse fisher matrix given the parameters

        :param param: rate parameters of poisson
        :return: inverse fisher matrix. Dim: belief_batch x state_dim x state_dim
        """

        return torch.diag_embed(param)

    def fisher_matrix(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
        """
               Computes the fisher matrix given the parameters

               :param param: rate parameters of poisson
               :return: fisher matrix. Dim: belief_batch x state_dim x state_dim
               """

        return torch.diag_embed(1. / param)

    def operator_grad_log(self, state, action, param):
        """
        Computes the master equation operator applied to the grad log of the prob.
        This is needed for the ODE that describes the parameters.
        The method needs the 'rates' method of the transition model to return rates and next_states.

        :param param: current belief of parameter
        :param state: state, here batch for all possible states is used
        :param action: action signal
        :return: dim action_batch x belief_batch x state_batch x param_dim
        """
        rates, next_states = self.t_model.rates(state, action)

        change_vectors = next_states - state
        # change_vectors Dimension num_events x action_batch x belief_batch x state_batch x state_dim
        sum = (change_vectors * rates[..., None]).sum(dim=0)
        # sum over the events. for poisson state_dim is the same as param_dim
        return sum / param

    def log_prob(self, state, param):
        """
        Returns the log of the pmf of the model, given parameter and state

        :param param: parameter (rates)
        :return: log probabilties, dim belief_batch x state_batch
        """
        #self.distribution.rate=param
        #q = self.distribution.log_prob(state)
        q = torch.distributions.Poisson(param).log_prob(state)
        return q.sum(-1)

    def grad_log_prob(self, state: torch.Tensor, param: torch.Tensor,**kwargs):
        """
        Returns the gradient of the log probability wrt param, evaluated at state
        :param state: state, where the grad_log_prob is evaluated
        :param param: rates of the poisson distribution
        :return: gradient of the log probability
        """

        batch_size = torch.broadcast_shapes(state.shape, param.shape)
        param = param.expand(*batch_size)

        return state / param - 1

    def sample(self, param, num_samples=1):
        #self.distribution.rate=param
        #state = self.distribution.sample((num_samples,))
        state = torch.distributions.Poisson(param).sample((num_samples,))
        return state.permute(*range(1, len(param.shape[:-1]) + 1), 0, -1)

    def moments(self, param):
        mean = param
        var = param
        skew = 1 / torch.sqrt(param)
        return mean, var, skew

    def drift(self, belief, action, **kwargs):
        """
        Overwriting ProjectionFilter.drift, as the inverse of the fisher matrix can be computed analytically.
        This function therefore calls the inverse directly instead of calling torch,solve(fisher_matrix...)
        Compute the right hand side of the ODE for the parameter.
        :param belief:
        :param action:
        :param kwargs:
        :return:
        """
        # If not called otherwise, use filters expectation method
        try:
            method = kwargs['method']
        except KeyError:
            method = self.exp_method
        try:
            num_samples = kwargs['num_samples']
        except KeyError:
            num_samples = self.exp_samples

        # dim before permute: action_batch x state_dim x belief_batch
        # output dim action_batch x belief_batch x state_dim
        def func2(state):
            return self.increment(state, action[..., None, :], belief[..., None, :], method=method).permute(-1, *range(
                len(state.shape[:-1])))

        def func(state):
            return self.operator_grad_log(state, action[..., None, :], belief[..., None, :]).permute(-1, *range(
                len(state.shape[:-1])))

        ## To compare. this one does the fisher matrix inside the expectation, if increment(including fisher_matrix) can be written in closed form, one should overwrite drift
        # drift = self.expectation_state_func(func2, belief, method=method).permute(*range(1, len(belief.shape[:-1]) + 1),0)

        drift = (self.inv_fisher_matrix(belief, method=method, num_samples=num_samples) @
                 self.expectation_state_func(func, belief, method=method, num_samples=num_samples).permute(
                     *range(1, len(belief.shape[:-1]) + 1),
                     0)[..., None]).sum(dim=-1)

        return drift

    def jump_update(self, param, observation, action,**kwargs):
        try:
            method = kwargs['method']
        except KeyError:
            method = self.exp_method
        try:
            num_samples = kwargs['num_samples']
        except KeyError:
            num_samples = self.exp_samples

        def numfunc(state):
            return ((self.o_model.log_prob(state, action[..., None, :],
                                           observation[..., None, :])).exp()[None]*state.permute(-1,*range(0,len(state.shape[:-1]))))
        def denfunc(state):
            return (self.o_model.log_prob(state, action[..., None, :],observation[..., None, :])).exp()


        num=self.expectation_state_func(numfunc,param,method=method,num_samples=num_samples)
        den=self.expectation_state_func(denfunc,param,method=method,num_samples=num_samples)
        param_new=num/den
        param_new = param_new.permute(*range(1,len(param_new.shape)),0)

        ### get rid of nan. If there is one use old param for the update

        if param_new.isnan().sum() > 0:
            if param.dim() == 1:
                param_new[:] = param
            else:
                idx,_  =param_new.isnan().max(0)
                param_new[:,idx] = param[:,idx]
        if param_new.isinf().sum() > 0:
            if param.dim() == 1:
                param_new[:] = param
            else:
                idx, _ = param_new.isinf().max(0)
                param_new[:,idx] = param[:,idx]

        return param_new.clip_(min=EPS)


