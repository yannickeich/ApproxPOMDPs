import torch

from packages.model.components.agents.projection_agent import FiniteProjectionAgent, DiscreteProjectionAgent
from packages.model.components.filter.discrete_time_obs.discrete_spaces.advanced_discrete_families import \
    BetaBinomialFilter, GammaPoissonFilter, UnscentedExpPoissonFilter, UnscentedSigmoidBinomialFilter, \
    NeuralBinomialFilter
from packages.model.components.filter.discrete_time_obs.discrete_spaces.simple_discrete_families import BinomialFilter, \
    PoissonFilter
from packages.model.components.filter.discrete_time_obs.discrete_spaces.trunc_discrete_filter import TruncPoissonFilter, \
    TruncGammaPoissonFilter, TruncUnscentedExpPoissonFilter
from packages.model.components.filter.discrete_time_obs.discrete_spaces.closed_form_discrete_filters import BinomialQueueingFilter, PoissonCRNFilter
from packages.model.components.observation_model import DiscreteTimeObservationModel
from packages.model.components.transition_model import DiscreteTransitionModel, QueueingTransitionModel, CRNTransitionModel


class BinomialAgent(BinomialFilter, FiniteProjectionAgent):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix=None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using a Binomial Projection Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        BinomialFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        FiniteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                       initial_time=initial_time, advantage_net=advantage_net, Q_matrix=Q_matrix, Q_function = Q_function,normalization=normalization,exp_method=exp_method,
                                       exp_samples=exp_samples, device=device, sim_method=sim_method,
                                       sim_options=sim_options,
                                       jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class BinomialQueueingAgent(BinomialQueueingFilter,FiniteProjectionAgent):
    def __init__(self, t_model: QueueingTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix=None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using a Binomial Projection Filter for Queueing systems

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        BinomialQueueingFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        FiniteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                       initial_time=initial_time, advantage_net=advantage_net, Q_matrix=Q_matrix,Q_function = Q_function, normalization=normalization,exp_method=exp_method,
                                       exp_samples=exp_samples, device=device, sim_method=sim_method,
                                       sim_options=sim_options,
                                       jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)



class BetaBinomialAgent(BetaBinomialFilter, FiniteProjectionAgent):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using a Beta-Binomial Projection Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        BinomialFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        FiniteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                       initial_time=initial_time, advantage_net=advantage_net, Q_matrix=Q_matrix,Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                       exp_samples=exp_samples, device=device, sim_method=sim_method,
                                       sim_options=sim_options,
                                       jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class PoissonAgent(PoissonFilter, DiscreteProjectionAgent):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using a Poisson Projection Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        PoissonFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                               initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples, device=device,
                               sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        DiscreteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                         initial_time=initial_time, advantage_net=advantage_net, Q_matrix=Q_matrix,Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                         exp_samples=exp_samples, device=device, sim_method=sim_method,
                                         sim_options=sim_options,
                                         jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)



class PoissonCRNAgent(PoissonCRNFilter, DiscreteProjectionAgent):
    def __init__(self, t_model: CRNTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using a Poisson Projection Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        PoissonCRNFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                               initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples, device=device,
                               sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        DiscreteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                         initial_time=initial_time, advantage_net=advantage_net, Q_matrix=Q_matrix, Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                         exp_samples=exp_samples, device=device, sim_method=sim_method,
                                         sim_options=sim_options,
                                         jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class TruncPoissonAgent(TruncPoissonFilter, FiniteProjectionAgent):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None, normalization=1,exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using a Poisson Projection Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        TruncPoissonFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                               initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples, device=device,
                               sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        FiniteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                       initial_time=initial_time, advantage_net=advantage_net, Q_matrix=Q_matrix,Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                       exp_samples=exp_samples, device=device, sim_method=sim_method,
                                       sim_options=sim_options,
                                       jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class GammaPoissonAgent(GammaPoissonFilter, DiscreteProjectionAgent):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using a Gamma Poisson  Projection Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        GammaPoissonFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                    initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                    device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        DiscreteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                         initial_time=initial_time, advantage_net=advantage_net, Q_matrix= Q_matrix,Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                         exp_samples=exp_samples, device=device, sim_method=sim_method,
                                         sim_options=sim_options,
                                         jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class TruncGammaPoissonAgent(TruncGammaPoissonFilter, FiniteProjectionAgent):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None, Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using a Gamma Poisson  Projection Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """

        TruncGammaPoissonFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                    initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                    device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        FiniteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                       initial_time=initial_time, advantage_net=advantage_net, Q_matrix= Q_matrix,Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                       exp_samples=exp_samples, device=device, sim_method=sim_method,
                                       sim_options=sim_options,
                                       jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class UnscentedExpPoissonAgent(UnscentedExpPoissonFilter, DiscreteProjectionAgent):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using an UnscentedExpPoissonAgent Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        UnscentedExpPoissonFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                                initial_time=initial_time, exp_method=exp_method,
                                                exp_samples=exp_samples, device=device, sim_method=sim_method,
                                                sim_options=sim_options,
                                                jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        DiscreteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                         initial_time=initial_time, advantage_net=advantage_net, Q_matrix= Q_matrix,Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                         exp_samples=exp_samples, device=device, sim_method=sim_method,
                                         sim_options=sim_options,
                                         jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class TruncUnscentedExpPoissonAgent(TruncUnscentedExpPoissonFilter, FiniteProjectionAgent):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None, Q_function = None,normalization=1,exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using an UnscentedExpPoissonAgent Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        TruncUnscentedExpPoissonFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                                initial_time=initial_time, exp_method=exp_method,
                                                exp_samples=exp_samples, device=device, sim_method=sim_method,
                                                sim_options=sim_options,
                                                jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        FiniteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                       initial_time=initial_time, advantage_net=advantage_net, Q_matrix= Q_matrix,Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                       exp_samples=exp_samples, device=device, sim_method=sim_method,
                                       sim_options=sim_options,
                                       jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class UnscentedSigmoidBinomialAgent(UnscentedSigmoidBinomialFilter, FiniteProjectionAgent):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None,normalization=1, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """
        Agent Using an UnscentedSigmoidBinomial Filter

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial param for filter
        :param initial_time: initial time point for filter
        :param advantage_net: advantage network for selecting actions
        """
        UnscentedSigmoidBinomialFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                                initial_time=initial_time, exp_method=exp_method,
                                                exp_samples=exp_samples, device=device, sim_method=sim_method,
                                                sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        FiniteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                       initial_time=initial_time, advantage_net=advantage_net, Q_matrix=Q_matrix,Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                       exp_samples=exp_samples, device=device, sim_method=sim_method,
                                       sim_options=sim_options,
                                       jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)


class NeuralProjectionAgent(FiniteProjectionAgent, NeuralBinomialFilter):
    def __init__(self, t_model: DiscreteTransitionModel,
                 o_model: DiscreteTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, Q_matrix = None,Q_function = None,normalization=1, inference_net=None, exp_method='MC', exp_samples=1000, device=None,
                 sim_method=None, sim_options=None, jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}, action_trajectory = None):
        """

        :param t_model:
        :param o_model:
        :param initial_param:
        :param initial_time:

        """
        FiniteProjectionAgent.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                       initial_time=initial_time, advantage_net=advantage_net, Q_matrix=Q_matrix,Q_function = Q_function,normalization=normalization, exp_method=exp_method,
                                       exp_samples=exp_samples, device=device, sim_method=sim_method,
                                       sim_options=sim_options,
                                       jump_optim=jump_optim, jump_opt_iter=jump_opt_iter, jump_opt_options=jump_opt_options, action_trajectory=action_trajectory)

        NeuralBinomialFilter.__init__(self, t_model, o_model, initial_param, initial_time, inference_net=inference_net,
                                      exp_method=exp_method, exp_samples=exp_samples, device=device,
                                      sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)