import torch

from packages.model.components.agents.projection_agent import ContTimeProjectionAgent
from packages.model.components.filter.cont_time_obs.cont_time_filter import MultinomialMonoCRNFilter, \
    PoissonMonoCRNFilter, TruncPoissonMonoCRNFilter
from packages.model.components.observation_model import ExactContTimeObservationModel
from packages.model.components.transition_model import MultinomialMonoCRNTransitionModel, PoissonMonoCRNTransitionModel, \
    TruncPoissonMonoCRNTransitionModel


class MultinomialMonoCRNAgent(MultinomialMonoCRNFilter, ContTimeProjectionAgent):
    t_model: MultinomialMonoCRNTransitionModel
    o_model = ExactContTimeObservationModel

    def __init__(self, t_model: MultinomialMonoCRNTransitionModel,
                 o_model: ExactContTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        MultinomialMonoCRNFilter.__init__(self, t_model, o_model, initial_param, initial_time, exp_method=exp_method,
                                          exp_samples=exp_samples, device=device, sim_method=sim_method,
                                          sim_options=sim_options,
                                          jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options)
        ContTimeProjectionAgent.__init__(self, t_model, o_model, initial_param, initial_time, advantage_net, Q_matrix,Q_function = Q_function,normalization=normalization,
                                         exp_method=exp_method, exp_samples=exp_samples, device=device, sim_method=sim_method,
                                         sim_options=sim_options,
                                         jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory = action_trajectory)


class PoissonMonoCRNAgent(PoissonMonoCRNFilter, ContTimeProjectionAgent):
    t_model: PoissonMonoCRNTransitionModel
    o_model = ExactContTimeObservationModel

    def __init__(self, t_model: PoissonMonoCRNTransitionModel,
                 o_model: ExactContTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None, Q_function = None,normalization=1,exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        PoissonMonoCRNFilter.__init__(self, t_model, o_model, initial_param, initial_time, exp_method=exp_method,
                                          exp_samples=exp_samples, device=device, sim_method=sim_method,
                                          sim_options=sim_options,
                                          jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options)
        ContTimeProjectionAgent.__init__(self, t_model, o_model, initial_param, initial_time, advantage_net, Q_matrix,Q_function = Q_function,normalization=normalization,
                                         exp_method=exp_method, exp_samples=exp_samples, device=device, sim_method=sim_method,
                                         sim_options=sim_options,
                                         jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class TruncPoissonMonoCRNAgent(TruncPoissonMonoCRNFilter, ContTimeProjectionAgent):
    t_model: TruncPoissonMonoCRNTransitionModel
    o_model = ExactContTimeObservationModel

    def __init__(self, t_model:TruncPoissonMonoCRNTransitionModel,
                 o_model: ExactContTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        TruncPoissonMonoCRNFilter.__init__(self, t_model, o_model, initial_param, initial_time, exp_method=exp_method,
                                          exp_samples=exp_samples, device=device, sim_method=sim_method,
                                          sim_options=sim_options,
                                          jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options)
        ContTimeProjectionAgent.__init__(self, t_model, o_model, initial_param, initial_time, advantage_net, Q_matrix,Q_function = Q_function,normalization=normalization,
                                         exp_method=exp_method, exp_samples=exp_samples, device=device, sim_method=sim_method,
                                         sim_options=sim_options,
                                         jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory = action_trajectory)