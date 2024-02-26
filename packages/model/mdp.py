from abc import ABC, abstractmethod
import torch
import numpy as np
from packages.model.components.spaces import Space
from packages.model.components.transition_model import TransitionModel, DiscreteTransitionModel
from packages.model.components.reward_model import RewardModel
#from packages.model.components.agent import Agent
from packages.model.components.trajectories import ContinuousTimeTrajectory, DiscreteTimeTrajectory




class MDPAgent():
    def __init__(self, s_space: Space, a_space: Space, Q: torch.Tensor = None, action_trajectory = None):
        """
        Create an agent that either takes actions according to Q-value function or according to an already defined action trajectory.
        :param s_space:
        :param a_space:
        """
        self.s_space = s_space
        self.a_space = a_space
        self.Q = Q
        self.action_trajectory = action_trajectory

    def get_action(self,state: torch.Tensor,t) -> torch.Tensor:


        if self.action_trajectory:
            return torch.tensor([self.action_trajectory(t)[0]]).long()
        # get state index
        try:
            index = np.ravel_multi_index(state.numpy().T, self.s_space.cardinalities)
        except:
            index = (state.reshape(-1, state.shape[-1])[..., None, :] == self.s_space.elements).prod(-1).nonzero()[..., -1]

        value, action = self.Q[:,index].max(0)

        return torch.tensor([action])

class MDP(ABC):
    def __init__(self, a_space: Space, s_space: Space, t_model: TransitionModel,
                 r_model: RewardModel, discount: float):
        self.a_space = a_space
        self.s_space = s_space
        self.t_model = t_model
        self.r_model = r_model
        self.discount = discount

    # Action Space
    # State Space
    # Transition Model
    # Reward Model

    @abstractmethod
    def simulate(self, t_grid: torch.Tensor, agent, initial_state):
        raise NotImplementedError
    # Return trajectory


class DiscreteMDP(MDP):
    t_model: DiscreteTransitionModel

    def __init__(self, a_space: Space, s_space: Space, t_model: DiscreteTransitionModel,
                 r_model: RewardModel, discount: float):
        super().__init__(a_space, s_space, t_model, r_model, discount)

    def simulate(self, t_grid: torch.Tensor, agent: MDPAgent, initial_state: torch.Tensor):
        t = t_grid[0].clone()
        T = t_grid[-1].clone()

        t_state_vector = torch.tensor([t]).clone()
        state_vector = initial_state[None, ...].clone()

        t_action_vector = torch.tensor([t]).clone()
        action_vector = agent.get_action(state_vector[-1],t)[None, ...].clone()

        t_reward_vector = torch.tensor([t]).clone()
        reward_vector = self.r_model(state_vector[-1], action_vector[-1])[None, ...].clone()

        while t_state_vector[-1] <= T:
            max_exit_rate = self.t_model.max_exit_rate(state_vector[-1])
            min_waiting = torch.distributions.exponential.Exponential(max_exit_rate).sample()

            #t_obs_new = self.o_model.sample_times([t, (t + min_waiting).clip(max=T)])
            t_grid_new = t_grid[torch.logical_and(t < t_grid, t_grid <= (t + min_waiting))]

            #t_event_new, srt_idx = torch.sort(torch.cat((t_obs_new, t_grid_new)))
            #t_event_is_obs = torch.cat((torch.ones_like(t_obs_new), torch.zeros_like(t_grid_new)))[srt_idx]

            for idx, t_event in enumerate(t_grid_new):
                # sample new observation
                action = agent.get_action(state_vector[-1],t_event)
                t_action_vector = torch.cat((t_action_vector, t_event.clone()[None, ...]))
                action_vector = torch.cat((action_vector, action.clone()[None, ...]))

                t_reward_vector = torch.cat((t_reward_vector, t_event.clone()[None, ...]))
                reward_vector = torch.cat(
                        (reward_vector, self.r_model(state_vector[-1], action_vector[-1]).clone()[None, ...]))

                # if t_event_is_obs[idx]:
                #     obs = self.o_model.sample(state_vector[-1], action)
                #     # update agent
                #     agent.add_observation(obs, t_event)
                #
                #     t_obs_vector = torch.cat((t_obs_vector, t_event.clone()[None, ...]))
                #     obs_vector = torch.cat((obs_vector, obs.clone()[None, ...]))
                #
                # else:
                t_state_vector = torch.cat((t_state_vector, t_event.clone()[None, ...]))
                state_vector = torch.cat((state_vector, state_vector[-1].clone()[None, ...]))

            t += min_waiting
            if t > T:
                break

            #   waiting_time, new_state, observations, rewards = self._simulate_transition(t_start=t, state_start=state,agent)
            # Thinning
            action = agent.get_action(state_vector[-1],t)
            t_action_vector = torch.cat((t_action_vector, t.clone()[None, ...]))
            action_vector = torch.cat((action_vector, action.clone()[None, ...]))

            t_reward_vector = torch.cat((t_reward_vector, t.clone()[None, ...]))
            reward_vector = torch.cat(
                    (reward_vector, self.r_model(state_vector[-1], action_vector[-1]).clone()[None, ...]))

            if torch.rand(1) <= self.t_model.exit_rate(state_vector[-1], action) / max_exit_rate:
                next_state = self.t_model.sample_next_state(state_vector[-1], action)
                t_state_vector = torch.cat((t_state_vector, t.clone()[None, ...]))
                state_vector = torch.cat((state_vector, next_state.clone()[None, ...]))

        state_trajectory = ContinuousTimeTrajectory(t_state_vector, state_vector)
        action_trajectory = ContinuousTimeTrajectory(t_action_vector, action_vector,interp_kind='next')
        reward_trajectory = ContinuousTimeTrajectory(t_reward_vector, reward_vector, interp_kind='next')

        return state_trajectory,  action_trajectory, reward_trajectory


