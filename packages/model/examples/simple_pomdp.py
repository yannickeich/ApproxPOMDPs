from packages.model.pomdp import DiscretePOMDP
from packages.model.components.transition_model import TabularTransitionModel
from packages.model.components.spaces import FiniteDiscreteSpace
from packages.model.components.observation_model import PoissonObservationModel
from packages.model.components.reward_model import RewardModel
import torch


# Classes for a simple POMDP with two actions/ two states and two observations
#
class SimpleStateSpace(FiniteDiscreteSpace):
    def __init__(self):
        """
        State space of simple POMDP
        """
        super().__init__(cardinalities=(2,))


class SimpleActionSpace(FiniteDiscreteSpace):
    def __init__(self):
        """
        Action space of simple POMDP
        """
        super().__init__(cardinalities=(2,))


class SimpleTransitionModel(TabularTransitionModel):
    s_space: SimpleStateSpace
    a_space: SimpleActionSpace

    def __init__(self, s_space: SimpleStateSpace, a_space: SimpleActionSpace, transition_rate=torch.tensor([.1]),
                 action_transition_rate=torch.tensor([1.])):
        """
        Transition model for simple POMDP

        :param s_space: state space
        :param a_space: action space
        :param transition_rate: transition rate for any action
        :param action_transition_rate: additional transition rate for action 0
        """
        self.transition_rate = transition_rate
        self.action_transition_rate = action_transition_rate
        super().__init__(s_space=s_space, a_space=a_space)

    def rates(self, state: torch.Tensor, action: torch.Tensor):
        """
        rate function of simple POMDP

        :param state: current state
        :param action: current action
        :return: rates and corresponding next states
        """
        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])

        rates = (self.transition_rate + self.action_transition_rate * (action[..., 0] == 1).expand(batch_size))[
            None]  # Dimension num_events x batch_size

        next_state = \
        torch.logical_not(state, out=torch.empty_like(state, dtype=state.dtype)).expand(*batch_size, state.shape[-1])[
            None]

        return rates, next_state


class SimpleObservationSpace(FiniteDiscreteSpace):
    def __init__(self):
        """
        Observation space for simple POMDP
        """
        super().__init__(cardinalities=(2,))


class SimpleObservationModel(PoissonObservationModel):
    s_space: SimpleStateSpace
    a_space: SimpleActionSpace
    o_space: SimpleObservationSpace

    def __init__(self, s_space: SimpleStateSpace, a_space: SimpleActionSpace, o_space: SimpleObservationSpace,
                 obs_rate=torch.tensor([2.0]), success_prob=torch.tensor([.9])):
        """
        Discrete Time Observation Model where the true stae is observed with a success probability
        :param s_space: state space
        :param a_space: action space
        :param o_space: observation space
        :param obs_rate: observation rate
        :param success_prob: success probability for observing the true state
        """
        self.success_prob = success_prob
        super().__init__(s_space, a_space, o_space, obs_rate)

    def sample(self, state: torch.tensor, action: torch.Tensor):
        """
        sample new observation

        :param state: current state
        :param action: current action
        """

        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])

        flip_state = torch.rand_like(state.expand(*batch_size,state.shape[-1]).to(dtype=torch.float)) > self.success_prob
        obs = torch.logical_xor(state, flip_state, out=torch.empty_like(flip_state))  # dimension num_state_batch , num_action_batch


        return obs

    def log_prob(self, state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor):
        """
        returns log emission_prob
        :param state: state_batch x action_batch x observation_batch x state_dim
        :param action: state_batch x action_batch x observation_batch x action_dim
        :param observation: state_batch x action_batch x observation_batch x observation_dim
        :return:
        """

        log_prob = ((observation[...,0] == state[...,0]) * self.success_prob.log() + (
                observation[...,0] != state[...,0]) * (
                                1 - self.success_prob).log())  # dimension num_observation_batch, num_state_batch, num_action_batch

        return log_prob


class SimpleRewardModel(RewardModel):
    def __call__(self, state: torch.Tensor, action: torch.Tensor, **kwargs):
        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])

        # only state 1 gives reward

        reward = state[..., 0].expand(batch_size)  # dimensions num_state_batch, num_action_batch

        return reward


class SimplePOMDP(DiscretePOMDP):
    s_space: SimpleStateSpace
    a_space: SimpleActionSpace
    o_space: SimpleObservationSpace
    t_model: SimpleTransitionModel
    o_model: SimpleObservationModel
    r_model: SimpleRewardModel

    def __init__(self, discount=1/-torch.log(torch.tensor(0.9))):
        a_space = SimpleActionSpace()
        s_space = SimpleStateSpace()
        o_space = SimpleObservationSpace()

        t_model = SimpleTransitionModel(s_space=s_space, a_space=a_space)
        o_model = SimpleObservationModel(s_space, a_space, o_space)
        r_model = SimpleRewardModel()
        super().__init__(a_space=a_space, s_space=s_space, o_space=o_space, t_model=t_model, o_model=o_model,
                         r_model=r_model, discount=discount)