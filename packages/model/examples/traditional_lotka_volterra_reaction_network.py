from packages.model.components.spaces import DiscreteSpace, FiniteDiscreteSpace,Space
from packages.model.components.transition_model import CRNTransitionModel,DiscreteTransitionModel
from packages.model.components.reward_model import RewardModel, QuadraticRewardModel
from packages.model.components.observation_model import PoissonObservationModel, GaussianObservationModel
from packages.model.pomdp import DiscretePOMDP
from packages.model.mdp import DiscreteMDP
import torch


class LotkaVolterraSSpace(DiscreteSpace):
    def __init__(self):
        """
        State space for a LotkaVolterra CRN

        """
        super().__init__(2)


class LotkaVolterraASpace(FiniteDiscreteSpace):
    def __init__(self):
        """
        Action space for the LotkaVolterra CRN (Death rate of a species can be controlled with 0 or 1)
        """
        super().__init__(cardinalities=(2,))



class LotkaVolterraOSpace(Space):
    def __init__(self):
        """
        Observation space of the Lotka Volterra CRN
        """

        super().__init__(False,2)







class LotkaVolterraPOMDP(DiscretePOMDP):
    s_space: LotkaVolterraSSpace
    a_space: LotkaVolterraASpace
    o_space: LotkaVolterraOSpace
    t_model: CRNTransitionModel
    o_model: GaussianObservationModel
    r_model: QuadraticRewardModel

    def __init__(self,
                 S: torch.Tensor = torch.tensor([[1, 0], [1, 1], [0, 1],[1, 0]]),
                 P: torch.Tensor = torch.tensor([[2, 0], [0, 2], [0, 0],[0, 0]]),
                 c: torch.Tensor = 2.5 * torch.tensor([[1.0, 0.01,0.50,  0.0], [1.0, 0.01, 1.0, 0.0]]).T,
                 obs_rate=torch.tensor([2.0]),goal = torch.tensor([50.,50.]),scale = 100,
                 linear = torch.eye(2), variance = torch.eye(2),
                 discount=1/-torch.log(torch.tensor(0.9)),device=None):
        """
        3D queueing network POMDP

        :param buffer_sizes: sizes of the three buffers
        :param arrival_rates: arrival rates for queue 0 and queue 1
        :param service_rates: service rates for queue 0, queue 1 and queue 2
        :param obs_rate:  Poisson rate parameter for discrete time observation
        :param success_prob: Binomial success probability for observing a packet in queue 2
        :param discount: POMDP discount
        """
        s_space = LotkaVolterraSSpace()
        a_space = LotkaVolterraASpace()
        o_space = LotkaVolterraOSpace()

        t_model = CRNTransitionModel(s_space=s_space, a_space=a_space,S=S.to(device),P=P.to(device),c=c.to(device))
        o_model = GaussianObservationModel(s_space=s_space, a_space=a_space, o_space=o_space, obs_rate=obs_rate.to(device),
                                                linear= linear, variance=variance)
        r_model = QuadraticRewardModel(goal= goal,scale =scale)

        super().__init__(a_space, s_space, o_space, t_model, o_model, r_model, discount)



class LotkaVolterraMDP(DiscreteMDP):
    s_space: LotkaVolterraSSpace
    a_space: LotkaVolterraASpace
    t_model: CRNTransitionModel
    r_model: QuadraticRewardModel

    def __init__(self, S: torch.Tensor = torch.tensor([[1, 0], [1, 1], [0, 1], [1, 0]]),
                 P: torch.Tensor = torch.tensor([[2, 0], [0, 2], [0, 0], [0, 0]]),
                 c: torch.Tensor = torch.tensor([[0.5, 0.05, 0.5, 0.0], [0.5, 0.05, 0.5, 1.0]]).T, goal = torch.tensor([10.,10.]),
                 discount=1/-torch.log(torch.tensor(0.9)),device=None):
        """
        3D queueing network MDP

        :param buffer_sizes: sizes of the three buffers
        :param arrival_rates: arrival rates for queue 0 and queue 1
        :param service_rates: service rates for queue 0, queue 1 and queue 2
        :param discount: MDP discount
        """
        s_space = LotkaVolterraSSpace()
        a_space = LotkaVolterraASpace()

        t_model = CRNTransitionModel(s_space=s_space, a_space=a_space,S=S.to(device),P=P.to(device),c=c.to(device))
        r_model = QuadraticRewardModel(goal=goal)

        super().__init__(a_space, s_space, t_model, r_model, discount)

