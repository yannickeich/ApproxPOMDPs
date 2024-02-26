from packages.model.components.spaces import DiscreteSpace, FiniteDiscreteSpace, MultinomialSpace
from packages.model.components.transition_model import MultinomialMonoCRNTransitionModel, PoissonMonoCRNTransitionModel, TruncPoissonMonoCRNTransitionModel
from packages.model.components.reward_model import RewardModel
from packages.model.components.observation_model import ExactContTimeObservationModel
from packages.model.pomdp import DiscreteStateContTimePOMDP
from packages.model.mdp import DiscreteMDP
import torch



##Network with X1,X2,X3,X4

class ContTime4DOSpace(DiscreteSpace):
    def __init__(self):
        """
        Observation space of the Lotka Volterra CRN
        """

        super().__init__(2)


class MonoCRNObservationModel(ExactContTimeObservationModel):
    s_space: DiscreteSpace
    a_space: FiniteDiscreteSpace
    o_space: DiscreteSpace

    def __call__(self, state: torch.Tensor):
        #returns the observed states
        obs_start = self.s_space.dimensions - self.o_space.dimensions
        return state[...,obs_start:]



class ContTime4DRewardModel(RewardModel):
    def __call__(self, state: torch.Tensor, action: torch.Tensor,**kwargs):
        """
        reward model for the ContTime4D network --> state in equilibrium -> punish maximum state

        :param state: 3d queueing state
        :param action: action signal
        :return: negative number of all packets
        """
        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        max_state,_=state.max(dim=-1)
        reward = - max_state

        return reward.expand(batch_size)

class MonoCRNRewardModel(RewardModel):
    def __call__(self, state: torch.Tensor, action: torch.Tensor,**kwargs):
        """
        reward model for the ContTime4D network --> state in equilibrium -> punish maximum state

        :param state: 3d queueing state
        :param action: action signal
        :return: negative number of all packets
        """
        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        mean_state = state.double().mean(dim=-1)
        reward = - torch.sqrt(((state.double() - mean_state[...,None])**2).sum(-1))

        return reward.expand(batch_size)



class MultinomialMonoCRNPOMDP(DiscreteStateContTimePOMDP):
    def __init__(self,c:torch.Tensor,obs_dim:int,total_N,discount=1/-torch.log(torch.tensor(0.9)),device=None):
        s_space = MultinomialSpace(total_N=total_N,dimensions = c.shape[1])
        a_space = FiniteDiscreteSpace((c.shape[0],))
        o_space = DiscreteSpace(dimensions=obs_dim)

        t_model = MultinomialMonoCRNTransitionModel(s_space=s_space, a_space=a_space, c=c.to(device), total_N=total_N)
        o_model = MonoCRNObservationModel(s_space=s_space, a_space=a_space, o_space=o_space)
        r_model = MonoCRNRewardModel()

        super().__init__(a_space=a_space, s_space=s_space, o_space=o_space, t_model=t_model, o_model=o_model, r_model=r_model, discount=discount)


class MultinomialMonoCRNMDP(DiscreteMDP):
    def __init__(self,c:torch.Tensor,total_N,discount=1/-torch.log(torch.tensor(0.9)),device=None):
        s_space = MultinomialSpace(total_N=total_N,dimensions = c.shape[1])
        a_space = FiniteDiscreteSpace((c.shape[0],))

        t_model = MultinomialMonoCRNTransitionModel(s_space=s_space, a_space=a_space, c=c.to(device), total_N=total_N)

        r_model = MonoCRNRewardModel()

        super().__init__(a_space=a_space, s_space=s_space, t_model=t_model,r_model=r_model,discount=discount)

class PoissonMonoCRNPOMDP(DiscreteStateContTimePOMDP):
    def __init__(self,c:torch.Tensor,b:torch.Tensor,d:torch.Tensor,obs_dim:int,discount=1/-torch.log(torch.tensor(0.9)),device=None):
        s_space = DiscreteSpace(dimensions = c.shape)
        a_space = FiniteDiscreteSpace((c.shape[0],))
        o_space = DiscreteSpace(dimensions=obs_dim)

        t_model = PoissonMonoCRNTransitionModel(s_space=s_space, a_space=a_space, c=c.to(device),b=b.to(device),d=d.to(device))
        o_model = MonoCRNObservationModel(s_space=s_space, a_space=a_space, o_space=o_space)
        r_model = MonoCRNRewardModel()

        super().__init__(a_space=a_space, s_space=s_space, o_space=o_space, t_model=t_model, o_model=o_model, r_model=r_model, discount=discount)


class PoissonMonoCRNMDP(DiscreteMDP):
    def __init__(self,c:torch.Tensor,b:torch.Tensor,d:torch.Tensor,discount=1/-torch.log(torch.tensor(0.9)),device=None):
        s_space = DiscreteSpace(dimensions = c.shape)
        a_space = FiniteDiscreteSpace((c.shape[0],))

        t_model = PoissonMonoCRNTransitionModel(s_space=s_space, a_space=a_space, c=c.to(device),b=b.to(device),d=d.to(device))
        r_model = ContTime4DRewardModel()

        super().__init__(a_space=a_space, s_space=s_space, t_model=t_model, r_model=r_model, discount=discount)


class TruncPoissonMonoCRNPOMDP(DiscreteStateContTimePOMDP):
    def __init__(self,truncation,c:torch.Tensor,b:torch.Tensor,d:torch.Tensor,obs_dim:int,discount=1/-torch.log(torch.tensor(0.9)),device=None):
        s_space = FiniteDiscreteSpace(truncation)
        a_space = FiniteDiscreteSpace((c.shape[0],))
        o_space = DiscreteSpace(dimensions=obs_dim)

        t_model = TruncPoissonMonoCRNTransitionModel(s_space=s_space, a_space=a_space, c=c.to(device),b=b.to(device),d=d.to(device))
        o_model = MonoCRNObservationModel(s_space=s_space, a_space=a_space, o_space=o_space)
        r_model = ContTime4DRewardModel()

        super().__init__(a_space=a_space, s_space=s_space, o_space=o_space, t_model=t_model, o_model=o_model, r_model=r_model, discount=discount)


class TruncPoissonMonoCRNMDP(DiscreteMDP):
    def __init__(self,truncation,c:torch.Tensor,b:torch.Tensor,d:torch.Tensor,discount=1/-torch.log(torch.tensor(0.9)),device=None):
        s_space = FiniteDiscreteSpace(truncation)
        a_space = FiniteDiscreteSpace((c.shape[0],))

        t_model = TruncPoissonMonoCRNTransitionModel(s_space=s_space, a_space=a_space, c=c.to(device),b=b.to(device),d=d.to(device))
        r_model = ContTime4DRewardModel()

        super().__init__(a_space=a_space, s_space=s_space, t_model=t_model, r_model=r_model, discount=discount)



