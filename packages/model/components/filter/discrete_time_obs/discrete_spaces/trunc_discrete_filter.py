import math
from typing import Tuple

import torch

from packages.model.components.filter.discrete_time_obs.discrete_spaces.discrete_parametric_filter import FiniteProjectionFilter, EPS
from packages.model.components.observation_model import DiscreteTimeObservationModel
from packages.model.components.transition_model import TruncCRNTransitionModel


class TruncPoissonFilter(FiniteProjectionFilter):
    t_model: TruncCRNTransitionModel
    o_model: DiscreteTimeObservationModel

    def belief_to_unc_belief(self, param):
        return torch.log(param)

    def unc_belief_to_belief(self, unc_param):
        return torch.exp(unc_param)

    def __init__(self, t_model: TruncCRNTransitionModel, o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor, initial_time: torch.Tensor, exp_method='MC', exp_samples=1000,
                 device=None, sim_method=None, sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Class for truncated poisson filter. Truncates the state space to a finite space (for example in sampling).
        T_model is a TruncCRNTransitionModel instead of CRNTransitionModel.
        Inherits from FiniteProjectionFilter instead of DiscreteProjectionFilter.

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial rate parameters for poisson distribution
        :param initial_time: initial time point
        """

        FiniteProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
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
        self.distribution.rate = param
        q = self.distribution.log_prob(state)
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
        self.distribution.rate=param
        state = self.distribution.sample((num_samples,))
        #Truncate the state
        state.clip(max=torch.tensor(self.t_model.s_space.cardinalities)-1)
        return state.permute(*range(1, len(param.shape[:-1]) + 1), 0, -1)

    def moments(self, param):
        mean = param
        var = param
        skew = 1 / torch.sqrt(param)
        return mean, var, skew

    def drift(self, belief, action, **kwargs):
        """
        Computes closed form solution for the drift of poisson parameters.
        :param belief: [belief_batch x belief_dim] or [belief_dim]
        :param action: dim [action_batch x 1] or [1]
        :param kwargs:
        :return: drift
        """
        #test = DiscreteProjectionFilter.drift(self, belief, action)
        batch_size = torch.broadcast_shapes(belief.shape[:-1], action.shape[:-1])
        # compare = super().drift(belief,action,**kwargs)
        S = self.t_model.S
        c = self.t_model.c[:, action].squeeze(-1)
        P = self.t_model.P
        S = S.expand(*batch_size, S.shape[0], S.shape[1]).permute(-2, *(range(len(batch_size))), -1)
        P = P.expand(*batch_size, P.shape[0], P.shape[1]).permute(-2, *(range(len(batch_size))), -1)
        # dims of P and S: [num_events, *batch_size, state_dim], dim c :[num_events x batch_size]
        change_vectors = P - S
        ### inner product (sum in log space) over the param dimenston, outer sum over the events
        return ((c * torch.exp((S * torch.log(belief[None, :]) - torch.lgamma(S + 1)).sum(-1)))[
                    ..., None] * change_vectors).sum(0)

    def jump_update(self, param, observation, action,**kwargs):
        try:
            method = kwargs['method']
        except KeyError:
            method = self.exp_method
        try:
            num_samples = kwargs['num_samples']
        except KeyError:
            num_samples = self.exp_samples

        if self.t_model.s_space.is_finite:
            method = 'exact'


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
            param_new[param_new.isnan()] = param[param_new.isnan()[0][None]]
        if param_new.isinf().sum() > 0:
            param_new[param_new.isinf()] = param[param_new.isinf()[0][None]]

        return param_new.clip_(min=EPS)

    def support_bounds(self) -> [torch.Tensor, torch.Tensor]:
        return torch.zeros_like(torch.tensor(self.t_model.s_space.cardinalities)), torch.tensor(self.t_model.s_space.cardinalities)-1


class TruncGammaPoissonFilter(FiniteProjectionFilter):
    t_model: TruncCRNTransitionModel
    o_model: DiscreteTimeObservationModel

    @staticmethod
    def split_param(param: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the param to alpha and beta param

        :param param: flattened param
        :return: alpha, beta (split params)
        """
        d = param.shape[-1]
        alpha = param[..., :int(d / 2)]
        beta = param[..., int(d / 2):]
        return alpha, beta

    @staticmethod
    def cat_param(param0: torch.Tensor, param1: torch.Tensor) -> torch.Tensor:
        """
        Concatenates alpha and beta parameters batchwise

        :param param0: alpha
        :param param1: beta
        :return: flattend param
        """
        return torch.cat((param0, param1), dim=-1)

    def belief_to_unc_belief(self, param):
        return param  # torch.log(param)

    def unc_belief_to_belief(self, unc_param):
        return unc_param.clip(min=EPS, max=1e6)  # torch.exp(unc_param).clip(min=EPS,max=1e10)

    def __init__(self, t_model: TruncCRNTransitionModel, o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor, initial_time: torch.Tensor, exp_method='MC', exp_samples=1000,
                 device=None, sim_method=None, sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Class for poisson filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial rate parameters for poisson distribution
        :param initial_time: initial time point
        """

        FiniteProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                          initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                          device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)

    def log_prob(self, state, param):
        """
        Returns the log of the pmf of the model, given parameter and state

        :param param: parameter (rates)
        :return: log probabilties, dim belief_batch x state_batch
        """
        alpha, beta = self.split_param(param)
        q = alpha * torch.log(beta) - torch.lgamma(alpha) - torch.lgamma(state + 1) + torch.lgamma(alpha + state) - (
                    alpha + state) * torch.log(beta + 1)
        return q.sum(-1)

    def grad_log_prob(self, state: torch.Tensor, param: torch.Tensor,**kwargs):
        """
        Returns the gradient of the log probability wrt param, evaluated at state
        :param state: state, where the grad_log_prob is evaluated
        :param param: alpha, beta, of the gamma distribution, that describe the distribution over rates parameter
        :return: gradient of the log probability
        """
        # comp=super().grad_log_prob(state,param)
        alpha, beta = self.split_param(param)
        grad_alpha_log_prob = torch.log(beta) - torch.digamma(alpha) + torch.digamma(alpha + state) - torch.log(
            beta + 1)
        grad_beta_log_prob = alpha / beta - (alpha + state) / (beta + 1)

        return torch.cat((grad_alpha_log_prob, grad_beta_log_prob), dim=-1)

    def sample(self, param, num_samples=1):
        alpha, beta = self.split_param(param)
        rate = torch.distributions.Gamma(alpha, beta).sample((num_samples,))
        state = torch.distributions.Poisson(rate).sample().clip(max=torch.tensor(self.t_model.s_space.cardinalities)-1)
        return state.permute(*range(1, len(param.shape[:-1]) + 1), 0, -1)

    def moments(self, param):
        raise NotImplementedError
    def support_bounds(self) -> [torch.Tensor, torch.Tensor]:
        return torch.zeros_like(torch.tensor(self.t_model.s_space.cardinalities)), torch.tensor(self.t_model.s_space.cardinalities)-1


class TruncUnscentedExpPoissonFilter(FiniteProjectionFilter):
    t_model: TruncCRNTransitionModel
    o_model: DiscreteTimeObservationModel

    @staticmethod
    def split_param(param: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the param to mean and covariance param

        :param param: flattened param
        :return: mean, cov_chol (split params)
        """
        batch_size = param.shape[:-1]
        d = int(-3 / 2 + math.sqrt(9 / 4 + 2 * param.shape[-1]))
        param0, param1 = torch.split(param, [d, int(d * (d + 1) / 2)], dim=-1)

        tril_indices = torch.tril_indices(d, d)
        param1_new = torch.zeros(*batch_size, d, d, device=param1.device, dtype=param1.dtype)
        param1_new[..., tril_indices[0], tril_indices[1]] = param1
        return param0, param1_new

    @staticmethod
    def cat_param(param0: torch.Tensor, param1: torch.Tensor) -> torch.Tensor:
        """
        Concatenates mean and covariance  parameters batchwise

        :param param0: mean parameter
        :param param1: cholesky covaraince parameter
        :return: flattend param
        """
        # param = torch.cat([param0, param1.flatten(start_dim=-2)], dim=-1)
        d = param1.shape[-1]
        param = torch.cat([param0, param1[...,torch.tril_indices(d, d)[0],torch.tril_indices(d,d)[1]]], dim=-1)
        return param

    def moments(self, param: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def belief_to_unc_belief(self, param):
        return param  # torch.log(param)

    def unc_belief_to_belief(self, unc_param):
        mean, cov = self.split_param(unc_param)
        #clips prob rate to (0.0025,0.9975)
        # mean = mean.clip(min=-6.0,max=6.0)
        param = self.cat_param(mean,cov)
        return param

    # TODO: comment
    def __init__(self, t_model: TruncCRNTransitionModel, o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor, initial_time: torch.Tensor, lambda_: float = 1.0, kappa: float = 1.0,
                 exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None, jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Class for binomial filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial success probabilties of binomials and lower triagonal of cholesky
        :param initial_time: initial time point
        """

        FiniteProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                        initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                        device=device, sim_method=sim_method, sim_options=sim_options,
                                        jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)



        self.lambda_ = lambda_
        self.kappa = kappa
        self.dim = self.t_model.s_space.dimensions
        w_0 = torch.tensor([self.lambda_ / (self.dim + self.kappa)])
        w_i = torch.tensor([1. / (2 * (self.dim + self.kappa))])
        self.weights = torch.cat([w_0, w_i.repeat(2 * self.dim)]).to(self.device)
        self.abscissas = torch.cat(
                [torch.zeros(self.dim)[None], math.sqrt(self.dim + self.lambda_) * torch.eye(self.dim),
                 -math.sqrt(self.dim + self.lambda_) * torch.eye(self.dim)]).to(self.device)

    def log_prob(self, state, param):
        """
        Returns the log of the pmf of the model, given parameter and state

        :param param: parameter (mean and cholesky elements of covariance (diag is in log space))
        :return: log probabilties, dim belief_batch x state_batch
        """
        # cov=cov_cho@torch.transpose(cov_cho,-1,-2)
        mean, cov_cho = self.split_param(param)
        cov_cho.diagonal(dim1=-1, dim2=-2).exp_()
        sigma_points = mean[..., None, :] + (cov_cho[..., None, :, :] @ self.abscissas[...,None]).sum(-1)
        log_poisson_prob = torch.distributions.Poisson( torch.exp(sigma_points)).log_prob(
                state[..., None, :]).sum(-1)
        log_prob = torch.logsumexp(log_poisson_prob + self.weights.log(), -1)

        return log_prob

    def sample(self, param, num_samples=1):
        # Output Dimension: (param.shape[:-1], num_samples, state_dim)

        # What I changed: sample N components and then for each one state, instead of sampling one components and using it for N states ( as in Mixture Distr)
        mean, cov_cho = self.split_param(param)

        cov_cho.diagonal(dim1=-1, dim2=-2).exp_()
        component = torch.distributions.Categorical(probs=self.weights).sample((*param.shape[:-1], num_samples,))

        # Adding sample dimension to mean and cov_cho. And adding dimension for abcissas for matrix vector product.
        sigma_point = mean[..., None, :] + (cov_cho[..., None, :, :] @ self.abscissas[component][..., None]).sum(-1)
        state = torch.distributions.Poisson(torch.exp(sigma_point)).sample().clip(max=torch.tensor(self.t_model.s_space.cardinalities)-1)

        return state

    def support_bounds(self) -> [torch.Tensor, torch.Tensor]:
        return torch.zeros_like(torch.tensor(self.t_model.s_space.cardinalities)), torch.tensor(self.t_model.s_space.cardinalities)-1