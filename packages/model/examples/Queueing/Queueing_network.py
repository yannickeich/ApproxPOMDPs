from packages.model.components.spaces import FiniteDiscreteSpace
from packages.model.components.transition_model import DiscreteTransitionModel, QueueingTransitionModel
from packages.model.components.reward_model import RewardModel
from packages.model.components.observation_model import PoissonObservationModel, GaussianObservationModel
from packages.model.pomdp import DiscretePOMDP
from packages.model.mdp import DiscreteMDP
import torch

#TODO fix observation space in POMDP and MDP

### Class for queueing networks



class QueueingStateSpace(FiniteDiscreteSpace):
    def __init__(self, buffer_sizes: torch.Tensor = torch.tensor([50, 50, 50])):
        """
        State space for a queueing network

        :param buffer_sizes: sizes of the buffers
        """
        self.buffer_sizes = buffer_sizes
        super().__init__(cardinalities=tuple((self.buffer_sizes + 1).tolist()))


class QueueingActionSpace(FiniteDiscreteSpace):
    def __init__(self,n_actions:int=2):
        """
        Action space for a queueing network

        :param n_actions: Number of actions
        """
        super().__init__(cardinalities=(n_actions,))


## for QueueingTransitionModel, see TransitionModel

class QueueingObservationSpace(FiniteDiscreteSpace):
    def __init__(self, obs_buffer_sizes: torch.Tensor = torch.tensor([50, 50])):
        """
        Observation space of a queuing_network

        :param obs_buffer_sizes: observation buffer sizes
        """
        self.obs_buffer_sizes = obs_buffer_sizes
        super().__init__(cardinalities=tuple((self.obs_buffer_sizes + 1).tolist()))


#ObservationModel can be either custom made or Gaussian



class QuadraticQueueingRewardModel(RewardModel):
    def __init__(self, goal,scale = 1):
        """
        Quadratic Reward Model for Queueing system.
        Has functions to compute the reward, given the state of the system
        and the expected reward given the belief.

        :param goal: goal state for the system
        :param scale: Scale for the reward model
        """
        self.goal = goal
        self.scale = scale
    def __call__(self,state,action,**kwargs):
        """

        :param state:
        :param action:
        :param kwargs:
        :return:
        """
        return - (((state - self.goal)/self.scale) ** 2).mean(-1)

    def expected_reward(self,filter, belief, action: torch.Tensor, **kwargs):
        """
        This computes the expected reward under product binomial distribution given the belief.
        The reward is the negative quadratic distance to the goal : R = - ((x-goal)/scale)**2.
        Given the belief, the expectation leads to: R_exp = - (n*p*(1-p) + (n*p)**2 - 2*n*p*goal + goal **2)/scale**2,
        where n is the maximal buffer size and p is the success probability which is described by belief.

        :param filter:
        :param belief: binomial success probability parameters of the belief distribution
        :param action:
        :return: expected reward
        """

        expected_reward = -(belief * filter.total_counts * (1 - belief) + (
                    filter.total_counts * belief) ** 2 - 2 * filter.total_counts * belief * self.goal + self.goal ** 2).sum(
            -1)
        return expected_reward/self.scale**2


class QueueingPOMDP(DiscretePOMDP):
    s_space: QueueingStateSpace
    a_space: QueueingActionSpace
    o_space: QueueingObservationSpace
    t_model: QueueingTransitionModel
    o_model: GaussianObservationModel
    r_model: QuadraticQueueingRewardModel

    def __init__(self, buffer_sizes: torch.Tensor = torch.tensor([1000, 1000, 1000]),
                 birth: torch.Tensor = torch.tensor([[10.0, 10.0, 0.0], [10.0, 10.0, 0.0]]),
                 death: torch.Tensor = torch.tensor([[0.0,0.0,20.0],[0.0,0.0,20.0]]),
                 inter_rates: torch.Tensor = torch.tensor([[[0.0, 0.0, 20.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                                           [[0.0, 0.0, 0.0], [0.0, 0.0, 20.0], [0.0, 0.0, 0.0]]]),
                 linear: torch.Tensor = torch.tensor([[0.,1.,0.],[0.,0.,1.]]),
                 variance: torch.Tensor = 5*torch.eye(2),
                 obs_rate=torch.tensor([2.0]), goal = torch.tensor([0.,0.,0.]), scale =100,
                 discount=5.0, device=None):
        """

        :param buffer_sizes:
        :param birth:
        :param death:
        :param inter_rates:
        :param linear:
        :param variance:
        :param obs_rate:
        :param goal:
        :param scale:
        :param discount:
        :param device:
        """

        assert (birth.shape == death.shape)
        assert (birth.shape == inter_rates.shape[:-1])

        s_space = QueueingStateSpace(buffer_sizes=buffer_sizes.to(device))
        a_space = QueueingActionSpace(n_actions=birth.shape[0])
        o_space = QueueingObservationSpace(obs_buffer_sizes=buffer_sizes[1:].to(device))

        t_model = QueueingTransitionModel(s_space=s_space, a_space=a_space, birth=birth, death=death, inter_rates=inter_rates)
        o_model = GaussianObservationModel(s_space=s_space, a_space=a_space, o_space=o_space, obs_rate=obs_rate.to(device),linear=linear,variance=variance)
        r_model = QuadraticQueueingRewardModel(goal=goal, scale = scale)

        super().__init__(a_space, s_space, o_space, t_model, o_model, r_model, discount)


class QueueingMDP(DiscreteMDP):
    s_space: QueueingStateSpace
    a_space: QueueingActionSpace
    t_model: QueueingTransitionModel
    r_model: QuadraticQueueingRewardModel

    def __init__(self, buffer_sizes: torch.Tensor = torch.tensor([1000, 1000, 1000]),
                 birth: torch.Tensor = torch.tensor([[10.0, 10.0, 0.0], [10.0, 10.0, 0.0]]),
                 death: torch.Tensor =torch.tensor([[0.0,0.0,20.0],[0.0,0.0,20.0]]),
                 inter_rates: torch.Tensor = torch.tensor([[[0.0, 0.0, 20.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                                           [[0.0, 0.0, 0.0], [0.0, 0.0, 20.0], [0.0, 0.0, 0.0]]]),
                 goal = torch.tensor([0.,0.,0.]), scale = 100,
                 discount=5.0, device=None):
        """
        3D queueing network MDP

        :param buffer_sizes: sizes of the three buffers
        :param arrival_rates: arrival rates for queue 0 and queue 1
        :param service_rates: service rates for queue 0, queue 1 and queue 2
        :param discount: MDP discount
        """
        assert (birth.shape == death.shape)
        assert (birth.shape == inter_rates.shape[:-1])

        s_space = QueueingStateSpace(buffer_sizes=buffer_sizes.to(device))
        a_space = QueueingActionSpace(n_actions = birth.shape[0])

        t_model = QueueingTransitionModel(s_space=s_space, a_space=a_space, birth=birth, death=death, inter_rates=inter_rates)
        r_model = QuadraticQueueingRewardModel(goal=goal,scale = scale)

        super().__init__(a_space, s_space, t_model, r_model, discount)