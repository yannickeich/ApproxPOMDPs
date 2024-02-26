from packages.model.components.spaces import FiniteDiscreteSpace, Space, ContinuousSpace
from packages.model.components.transition_model import DiscreteTransitionModel, TabularTransitionModel, \
    ContinuousTransitionModel
from packages.model.components.reward_model import RewardModel
from packages.model.components.observation_model import PoissonObservationModel, GaussianObservationModel
from packages.model.pomdp import DiscretePOMDP, ContinuousPOMDP
from packages.model.mdp import DiscreteMDP
import torch


##TODO make batchwise
## dx = Ax dt + Bu dt + c dw
class LQGTransitionModel(ContinuousTransitionModel):
    def __init__(self, s_space, a_space, A=torch.tensor([[1, 0], [0, 1]]), b=torch.tensor([[0], [1]]),
                 c=torch.tensor(0)):
        """
        Transition model for linear drift and dispersion
        :param s_space:
        :param a_space:
        :param A:
        :param b:
        """
        self.A = A
        self.b = b
        self.c = c
        super().__init__(s_space, a_space)

    def drift(self, state: torch.Tensor, action: torch.Tensor):
        action = action + 0.0
        return (self.A @ state[..., None] + self.b @ action[..., None]).squeeze(-1)

    def dispersion(self, state: torch.Tensor, action: torch.Tensor):
        batch_size = state.shape[:-1]
        state_dim = state.shape[-1]
        return self.c * torch.zeros(*batch_size, state_dim, state_dim)


class LQGRewardModel(RewardModel):
    def __call__(self, state: torch.Tensor, action: torch.Tensor, Q=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                 R=torch.tensor([[1.0]]), **kwargs):
        """
        reward model for LGQ
        :param state:
        :param action:
        :param kwargs:
        :return:
        """
        action = action + 0.0
        reward = - ((state[..., None].transpose(-2, -1) @ Q @ state[..., None]).sum(-1).sum(-1) + (
                    action[..., None].transpose(-2, -1) @ R @ action[..., None].transpose(-2, -1)).sum(-1).sum(-1))

        return reward


class LQGPOMDP(ContinuousPOMDP):
    s_space: ContinuousSpace
    a_space: Space
    o_space: ContinuousSpace
    t_model: LQGTransitionModel
    o_model: GaussianObservationModel
    r_model: LQGRewardModel

    def __init__(self, A=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), b=torch.tensor([[0.0], [1.0]]), c=torch.tensor([0.0]),
                 Q=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                 R=torch.tensor([1.0]), obs_rate=torch.tensor([20.0]),
                 variance=torch.tensor([[0.01, 0.0], [0.0, 0.01]]), discount=1/-torch.log(torch.tensor(0.9))):
        s_space = ContinuousSpace(2)
        a_space = FiniteDiscreteSpace(cardinalities=(3,))
        o_space = ContinuousSpace(2)

        t_model = LQGTransitionModel(s_space, a_space, A, b, c)
        o_model = GaussianObservationModel(s_space, a_space, o_space, obs_rate, variance)

        r_model = LQGRewardModel()

        super().__init__(a_space=a_space, s_space=s_space, o_space=o_space, t_model=t_model, o_model=o_model,
                         r_model=r_model, discount=discount)
