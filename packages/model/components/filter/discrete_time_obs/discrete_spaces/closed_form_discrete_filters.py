import torch

from packages.model.components.filter.discrete_time_obs.discrete_spaces.simple_discrete_families import BinomialFilter, PoissonFilter
from packages.model.components.observation_model import DiscreteTimeObservationModel, GaussianObservationModel
from packages.model.components.spaces import FiniteDiscreteSpace
from packages.model.components.transition_model import QueueingTransitionModel, CRNTransitionModel


#Make sure that it is the same in all files
EPS = 1e-4


class BinomialQueueingFilter(BinomialFilter):
    t_model = QueueingTransitionModel
    o_model = DiscreteTimeObservationModel

    def __init__(self, t_model: QueueingTransitionModel, o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor, initial_time: torch.Tensor, exp_method='MC', exp_samples=1000,
                 device=None, sim_method=None, sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Class for binomial filter if the transitio nmodel is a QueueingTransitionModel.
        Leads to closed form solutions for the drift.

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial success probabilties of binomials
        :param initial_time: initial time point
        """

        BinomialFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                        initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                        device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)

        assert isinstance(self.t_model.s_space, FiniteDiscreteSpace), "StateSpace should be Finite for BinomialFilter"
        self.total_counts = torch.tensor(self.t_model.s_space.cardinalities) - 1
        self.distribution = torch.distributions.Binomial(self.total_counts, initial_param)



    def jump_update(self, param, observation, action,**kwargs):
        """
        Updates the belief, given a new observation. If the ObservationModel is Gaussian,
        we approximate the current binomial Distribution by a Gaussian Distribution.
        If not Moment matching is done in the BinomialFilters jump update function.
        :param param:
        :param observation:
        :param action:
        :param kwargs:
        :return:
        """

        if isinstance(self.o_model, GaussianObservationModel):
            mean,var,_ = self.moments(param)
            variance_matrix = torch.zeros(*mean.shape,mean.shape[-1])
            variance_matrix[...,torch.arange(mean.shape[-1]),torch.arange(mean.shape[-1])] = var
            new_param = mean + (variance_matrix @ self.o_model.linear.T @ torch.linalg.inv(
                self.o_model.linear @ variance_matrix @ self.o_model.linear.T + self.o_model.variance) @ (
                                    observation - (self.o_model.linear @ mean[...,None]).sum(-1))[...,None]).sum(-1)
            new_param = 1 / self.total_counts * new_param
            return new_param.clip_(min=EPS, max=1 - EPS)

        return BinomialFilter.jump_update(self,param,observation, action, **kwargs)

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

        #test = BinomialFilter.drift(self, belief, action)

        birth_rates = self.t_model.birth_rates[action].squeeze(-2) #Squeeze over action_dim
        death_rates = self.t_model.death_rates[action].squeeze(-2)
        inter_rates = self.t_model.inter_rates[action].squeeze(-3)
        total_counts = self.total_counts

        drift = (1 - belief ** total_counts) / total_counts * (
                birth_rates + (inter_rates * (1 - (1 - belief) ** total_counts)[...,None]).sum(-2))
        drift += - (1 - (1 - belief) ** total_counts) / total_counts * (
                death_rates + (inter_rates.transpose(-1, -2) * (1 - belief ** total_counts)[..., None]).sum(-2))

        return drift




class PoissonCRNFilter(PoissonFilter):
    t_model: CRNTransitionModel
    o_model: DiscreteTimeObservationModel

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

        PoissonFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                          initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                          device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        self.distribution = torch.distributions.Poisson(initial_param)


    def drift(self, belief, action, **kwargs):
        """
        Computes closed form solution for the drift of poisson parameters.
        :param belief: [belief_batch x belief_dim] or [belief_dim]
        :param action: dim [action_batch x 1] or [1]
        :param kwargs:
        :return: drift
        """
        #test = PoissonFilter.drift(self, belief, action)
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


        if isinstance(self.o_model,GaussianObservationModel):
            new_param = param + torch.diag(param)@self.o_model.linear.T @ torch.linalg.inv(self.o_model.linear @ torch.diag(param) @ self.o_model.linear.T + self.o_model.variance)@(observation - self.o_model.linear @ param)
            return new_param.clip_(min=EPS)

        return PoissonFilter.jump_update(self, param, observation, action, **kwargs)


