# State Space
from packages.model.components.spaces import FiniteDiscreteSpace
from packages.model.components.transition_model import DiscreteTransitionModel , QueueingTransitionModel
from packages.model.components.reward_model import QuadraticRewardModel, RewardModel
from packages.model.components.observation_model import PoissonObservationModel, GaussianObservationModel
from packages.model.pomdp import DiscretePOMDP
from packages.model.mdp import DiscreteMDP
from packages.model.components.filter.discrete_time_obs.discrete_spaces.simple_discrete_families import PoissonFilter, BinomialFilter
import torch


# Classe for a simple queueing network
#
# structure: one queue
# actions: can either be service rate 1 or 2
# observations: gaussians
#rewards: quadratic + extra cost for using the second queue (action 1)


class Queueing1DSSpace(FiniteDiscreteSpace):
    def __init__(self, buffer_sizes: torch.Tensor = torch.tensor([10,])):
        """
        State space for a 1D queueing network

        :param buffer_sizes: sizes of the three buffers
        """
        self.buffer_sizes = buffer_sizes
        super().__init__(cardinalities=tuple((self.buffer_sizes + 1).tolist()))


class Queueing1DASpace(FiniteDiscreteSpace):
    def __init__(self):
        """
        Action space for the qD queueing network (use service rate 0 or 1)
        """
        super().__init__(cardinalities=(2,))



class Queueing1DOSpace(FiniteDiscreteSpace):
    def __init__(self, obs_buffer_sizes: torch.Tensor = torch.tensor([10,])):
        """
        Observation space of the queue

        :param obs_buffer_sizes: observation buffer size
        """
        self.obs_buffer_sizes = obs_buffer_sizes
        super().__init__(cardinalities=tuple((self.obs_buffer_sizes + 1).tolist()))


class Queueing1DRewardModel(QuadraticRewardModel):
    def __init__(self, goal, scale = 1,cost_action=torch.tensor(10.0) ):
        QuadraticRewardModel.__init__(self,goal=goal,scale=scale)
        self.cost_action = cost_action

    def __call__(self,state,action,**kwargs):
        return QuadraticRewardModel.__call__(self,state,action,) - 25 * (action[...,0]==1)

    def expected_reward(self,filter, belief, action: torch.Tensor, **kwargs):
        """
        Make sure that the action penalty is not added twice.
        Therefore it is added if exact computation of reward for QuadraticModel is available, but not for the ExpectationFunction, since it is included in the __call__ function.
        :param filter:
        :param belief:
        :param action:
        :param kwargs:
        :return:
        """
        if isinstance(filter,BinomialFilter):
            reward = QuadraticRewardModel.expected_reward(self,filter,belief,action,**kwargs) - 25 * (action[...,0]==1)
        else:
            reward = RewardModel.expected_reward(self, filter, belief, action, **kwargs)

        return reward


class Queueing1DPOMDP(DiscretePOMDP):
    s_space: Queueing1DSSpace
    a_space: Queueing1DASpace
    o_space: Queueing1DOSpace
    t_model: QueueingTransitionModel
    o_model: GaussianObservationModel
    r_model: Queueing1DRewardModel

    def __init__(self, buffer_sizes: torch.Tensor = torch.tensor([10,]),
                 birth: torch.Tensor = torch.tensor([[1.0,], [1.0, ]]),
                 death: torch.Tensor = torch.tensor([[1.0,],[2.0,]]),
                 rates: torch.Tensor = torch.zeros(2,1,1),
                 linear: torch.Tensor = torch.eye(1),
                 variance: torch.Tensor = torch.eye(1),
                 obs_rate=torch.tensor([2.0]), goal = torch.tensor([0.,]),
                 discount=5.0,device=None):
        """
        1D queueing network POMDP

        :param buffer_sizes: sizes of the three buffers
        :param arrival_rates: arrival rates for queue 0 and queue 1
        :param service_rates: service rates for queue 0, queue 1 and queue 2
        :param obs_rate:  Poisson rate parameter for discrete time observation
        :param success_prob: Binomial success probability for observing a packet in queue 2
        :param discount: POMDP discount
        """
        s_space = Queueing1DSSpace(buffer_sizes=buffer_sizes.to(device))
        a_space = Queueing1DASpace()
        o_space = Queueing1DOSpace(obs_buffer_sizes=buffer_sizes[1:].to(device))

        t_model = QueueingTransitionModel(s_space=s_space, a_space=a_space, birth=birth, death=death,
                                          inter_rates=rates)
        o_model = GaussianObservationModel(s_space=s_space, a_space=a_space, o_space=o_space, obs_rate=obs_rate.to(device),linear=linear,variance=variance)
        r_model = Queueing1DRewardModel(goal=goal)

        super().__init__(a_space, s_space, o_space, t_model, o_model, r_model, discount)


class Queueing1DMDP(DiscreteMDP):
    s_space: Queueing1DSSpace
    a_space: Queueing1DASpace
    t_model: QueueingTransitionModel
    r_model: Queueing1DRewardModel

    def __init__(self,  buffer_sizes: torch.Tensor = torch.tensor([10,]),
                 birth: torch.Tensor = torch.tensor([[1.0,], [1.0,]]),
                 death: torch.Tensor = torch.tensor([[1.0,],[2.0,]]),
                 rates: torch.Tensor = torch.zeros(2,1,1), goal = torch.tensor([0.,]),
        discount=5.0,device=None):
        """
        1D queueing network MDP

        :param buffer_sizes: sizes of the three buffers
        :param arrival_rates: arrival rates for queue 0 and queue 1
        :param service_rates: service rates for queue 0, queue 1 and queue 2
        :param discount: MDP discount
        """
        s_space = Queueing1DSSpace(buffer_sizes=buffer_sizes.to(device))
        a_space = Queueing1DASpace()

        t_model = QueueingTransitionModel(s_space=s_space, a_space=a_space, birth=birth, death=death,
                                          inter_rates=rates)
        r_model = Queueing1DRewardModel(goal=goal)

        super().__init__(a_space, s_space, t_model, r_model, discount)