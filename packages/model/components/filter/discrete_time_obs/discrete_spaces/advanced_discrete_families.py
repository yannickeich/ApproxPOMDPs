import math
from abc import ABC
from typing import Tuple

import torch
from torch import nn as nn

from packages.model.components.filter.discrete_time_obs.discrete_spaces.discrete_parametric_filter import FiniteProjectionFilter, EPS, \
    DiscreteProjectionFilter
from packages.model.components.observation_model import DiscreteTimeObservationModel
from packages.model.components.spaces import FiniteDiscreteSpace
from packages.model.components.transition_model import DiscreteTransitionModel, CRNTransitionModel


class BetaBinomialFilter(FiniteProjectionFilter, ABC):
    t_model: DiscreteTransitionModel
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
        return unc_param.clip(min=EPS, max=1e5)  # torch.exp(unc_param).clip(min=EPS,max=1e5)

    def __init__(self, t_model: DiscreteTransitionModel, o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor, initial_time: torch.Tensor, exp_method='MC', exp_samples=1000,
                 device=None, sim_method=None, sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Class for beta-binomial filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial alpha and beta parameters of beta distribution, that describe a dist over success probs p
        :param initial_time: initial time point
        """

        FiniteProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                        initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                        device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)

        assert isinstance(self.t_model.s_space,
                          FiniteDiscreteSpace), "StateSpace should be Finite for BetaBinomialFilter"
        self.total_counts = torch.tensor(self.t_model.s_space.cardinalities) - 1

    # def fisher_matrix(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
    #    # No exact solution for all entries, given --> Use the super() method. maybe compare speed. accuracy should be the same
    #     raise NotImplementedError

    def log_prob(self, state, param):
        """
        Returns the log of the pmf of the model, given parameter and state

        :param param: parameter (alpha, beta)
        :return: log probabilties, dim belief_batch x state_batch
        """
        alpha, beta = self.split_param(param)
        # TODO check dimenstions
        q = torch.lgamma(self.total_counts + 1) - torch.lgamma(state + 1) - torch.lgamma(self.total_counts - state + 1)
        q += torch.lgamma(state + alpha) + torch.lgamma(self.total_counts - state + beta) - torch.lgamma(
            self.total_counts + alpha + beta)
        q += torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        return q.sum(-1)

    def sample(self, param, num_samples=1):
        alpha, beta = self.split_param(param)
        # TODO check dimensions
        success_probs = torch.distributions.Beta(alpha, beta).sample((num_samples,))
        state = torch.distributions.Binomial(self.total_counts, success_probs).sample()
        return state.permute(*range(1, len(param.shape[:-1]) + 1), 0, -1)

    def hessian_log_prob(self, state: torch.Tensor, param: torch.Tensor, **kwargs):
        batch_size = torch.broadcast_shapes(state.shape[:-1], param.shape[:-1])
        alpha, beta = self.split_param(param)
        d = alpha.shape[-1]
        hessian = torch.zeros(*batch_size,2 * d, 2 * d)
        diagonal1 = torch.zeros(*batch_size,d)
        diagonal2 = torch.zeros(*batch_size,d)
        counts = self.total_counts
        ## TODO: Vectorize as much as possible
        # Maybe do first part with diag
        diagonal1 += - torch.polygamma(1, self.total_counts + alpha + beta) + torch.polygamma(1,
                                                                                             alpha + beta) - torch.polygamma(
            1, alpha)
        diagonal2 += - torch.polygamma(1, self.total_counts + alpha + beta) + torch.polygamma(1,
                                                                                             alpha + beta) - torch.polygamma(
            1, beta)


        diagonal1 += torch.polygamma(1, state + alpha)
        diagonal2 += torch.polygamma(1, counts - state + beta)
        hessian += torch.diag_embed(torch.cat((diagonal1, diagonal2), dim=-1))

        hessian += torch.diag_embed(-torch.polygamma(1, counts + alpha + beta) + torch.polygamma(1, alpha + beta),
                                          d) + torch.diag_embed(
            -torch.polygamma(1, counts + alpha + beta) + torch.polygamma(1, alpha + beta), d).transpose(-1, -2)
        return hessian



    def moments(self, param):
        alpha, beta = self.split_param(param)
        mean = self.total_counts * alpha / (alpha + beta)
        var = self.total_counts * alpha * beta * (self.total_counts + alpha + beta) / (
                    (alpha + beta) ** 2 * (alpha + beta + 1))
        skew = (alpha + beta + 2 * self.total_counts) * (beta - alpha) / (alpha + beta + 2) * torch.sqrt(
            (1 + alpha + beta) / (self.total_counts * alpha * beta * (self.total_counts + alpha + beta)))
        return mean, var, skew

    def support_bounds(self) -> [torch.Tensor, torch.Tensor]:
        return torch.zeros_like(self.total_counts), self.total_counts

    def jump_update(self, param, observation, action,**kwargs):
        return FiniteProjectionFilter.jump_update(self, param, observation, action).clip_(min=EPS)

    def grad_log_prob(self, state: torch.Tensor, param: torch.Tensor,**kwargs):
        ## TODO, check if faster than autograd(yes for single batch), but small num differences e-07, check dimensions
        # comp=super().grad_log_prob(state,param)
        alpha, beta = self.split_param(param)
        grad_alpha_log_prob = torch.digamma(state + alpha) - torch.digamma(
            self.total_counts + alpha + beta) + torch.digamma(alpha + beta) - torch.digamma(alpha)
        grad_beta_log_prob = torch.digamma(self.total_counts - state + beta) - torch.digamma(
            self.total_counts + alpha + beta) + torch.digamma(alpha + beta) - torch.digamma(beta)
        # return comp
        return torch.cat((grad_alpha_log_prob, grad_beta_log_prob), dim=-1)


class UnscentedSigmoidBinomialFilter(FiniteProjectionFilter, ABC):
    t_model: DiscreteTransitionModel
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
        mean = mean.clip(min=-6.0,max=6.0)
        param = self.cat_param(mean,cov)
        return param

    # TODO: comment
    def __init__(self, t_model: DiscreteTransitionModel, o_model: DiscreteTimeObservationModel,
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

        assert isinstance(self.t_model.s_space, FiniteDiscreteSpace), "StateSpace should be Finite for BinomialFilter"
        self.total_counts = torch.tensor(self.t_model.s_space.cardinalities) - 1
        self.lambda_ = lambda_
        self.kappa = kappa
        self.dim = len(self.total_counts)
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
        log_binomial_prob = torch.distributions.Binomial(self.total_counts, torch.sigmoid(sigma_points)).log_prob(
            state[..., None, :]).sum(-1)
        log_prob = torch.logsumexp(log_binomial_prob + self.weights.log(), -1)

        return log_prob

    def sample(self, param, num_samples=1):
        # Output Dimension: (param.shape[:-1], num_samples, state_dim)

        # What I changed: sample N components and then for each one state, instead of sampling one components and using it for N states ( as in Mixture Distr)
        mean, cov_cho = self.split_param(param)

        cov_cho.diagonal(dim1=-1, dim2=-2).exp_()
        component = torch.distributions.Categorical(probs=self.weights).sample((*param.shape[:-1], num_samples,))

        # Adding sample dimension to mean and cov_cho. And adding dimension for abcissas for matrix vector product.
        sigma_point = mean[..., None, :] + (cov_cho[..., None, :, :] @ self.abscissas[component][..., None]).sum(-1)
        state = torch.distributions.Binomial(self.total_counts, torch.sigmoid(sigma_point)).sample()

        return state

    def support_bounds(self) -> [torch.Tensor, torch.Tensor]:
        return torch.zeros_like(self.total_counts), self.total_counts

    # def grad_log_prob(self, state: torch.Tensor, param: torch.Tensor):
    #     # Overwrites the autograd version in projectionfilter with exact solution
    #     # Works, but is not faster than autograd
    #
    #
    #     batch_size = torch.broadcast_shapes(state.shape[:-1], param.shape[:-1])
    #     param = param.expand(*batch_size, param.shape[-1])
    #
    #     mean, cov_cho = self.split_param(param)
    #     cov_cho.diagonal(dim1=-1, dim2=-2).exp_()
    #     sigma_points = mean[...,None,:] + (cov_cho[...,None,:,:] * self.abscissas[:,None,:]).sum(-1)
    #
    #     #Compute the product of the binomials, by summing in the log_space
    #     weighted_binom = (self.weights.log() + torch.distributions.Binomial(self.total_counts,torch.sigmoid(sigma_points)).log_prob(state[...,None,:]).sum(-1)).exp()
    #     #sum over the unscented points
    #     norm = weighted_binom.sum(-1)
    #
    #     # add new dimension for the grad to the mean and sum over the unscented points
    #     grad_mean_log_prob = (weighted_binom[...,None] * (state[...,None,:] - self.total_counts * torch.sigmoid(sigma_points))).sum(-2)/norm[...,None]
    #
    #     # take lower trig of the following ...x d x d
    #     d = mean.shape[-1]
    #     tril_ind = torch.tril_indices(d, d)
    #
    #     # add 2 new dimension for the grad to the cov_cho and sum over the unscented points
    #     grad_cov_log_prob = ((weighted_binom[...,None] * (state[...,None,:] - self.total_counts * torch.sigmoid(sigma_points)))[...,None] * self.abscissas[:,None,:]).sum(-3)/norm[...,None,None]
    #
    #     # take lower triagonal
    #     d = mean.shape[-1]
    #     tril_ind = torch.tril_indices(d, d)
    #     grad_cov_log_prob = grad_cov_log_prob[...,tril_ind[0],tril_ind[1]]
    #
    #     return torch.cat((grad_mean_log_prob,grad_cov_log_prob),-1)


class NeuralBinomialFilter(FiniteProjectionFilter, ABC):
    t_model: DiscreteTransitionModel
    o_model: DiscreteTimeObservationModel

    def __init__(self, t_model: DiscreteTransitionModel, o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor, initial_time: torch.Tensor, inference_net: nn.Module, exp_method='MC',
                 exp_samples=1000, device=None, sim_method=None, sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
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
        self.inference_net = inference_net

    def belief_to_unc_belief(self, param):
        return torch.logit(param)

    def unc_belief_to_belief(self, unc_param):
        return torch.sigmoid(unc_param)

    def log_prob(self, state, param):
        """
        Returns the log of the pmf of the model, given parameter and state

        :param param: parameter (success probabilties)
        :return: log probabilties, dim belief_batch x state_batch
        """

        success_probs = self.inference_net(param)
        log_prob = torch.distributions.Binomial(self.total_counts, success_probs).log_prob(state).sum(-1)
        return log_prob

    def sample(self, param, num_samples=1):
        success_probs = self.inference_net(param)
        state = torch.distributions.Binomial(self.total_counts, success_probs).sample((num_samples,))
        return state.permute(*range(1, len(param.shape[:-1]) + 1), 0, -1)

    def support_bounds(self) -> [torch.Tensor, torch.Tensor]:
        return torch.zeros_like(self.total_counts), self.total_counts

    def moments(self, param: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GammaPoissonFilter(DiscreteProjectionFilter):
    t_model: CRNTransitionModel
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

    def __init__(self, t_model: CRNTransitionModel, o_model: DiscreteTimeObservationModel,
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
        state = torch.distributions.Poisson(rate).sample()
        return state.permute(*range(1, len(param.shape[:-1]) + 1), 0, -1)

    def moments(self, param):
        raise NotImplementedError


class UnscentedExpPoissonFilter(DiscreteProjectionFilter):
    t_model: CRNTransitionModel
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
    def __init__(self, t_model: DiscreteTransitionModel, o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor, initial_time: torch.Tensor, lambda_: float = 1.0, kappa: float = 1.0,
                 exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None, jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Class for binomial filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial success probabilties of binomials and lower triagonal of cholesky
        :param initial_time: initial time point
        """

        DiscreteProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
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
        state = torch.distributions.Poisson(torch.exp(sigma_point)).sample()

        return state


