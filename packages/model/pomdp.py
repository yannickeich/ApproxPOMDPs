from abc import ABC, abstractmethod
import torch
from packages.model.components.spaces import Space, DiscreteSpace, ContinuousSpace
from packages.model.components.transition_model import TransitionModel, DiscreteTransitionModel, \
    ContinuousTransitionModel
from packages.model.components.observation_model import ObservationModel, PoissonObservationModel, ExactContTimeObservationModel
from packages.model.components.reward_model import RewardModel
from packages.model.components.agents.agent import Agent
from packages.model.components.trajectories import ContinuousTimeTrajectory, DiscreteTimeTrajectory
from packages.model.mdp import DiscreteMDP


# TODO Document

class POMDP(ABC):
    def __init__(self, a_space: Space, s_space: Space, o_space: Space, t_model: TransitionModel,
                 o_model: ObservationModel, r_model: RewardModel, discount: float):
        self.a_space = a_space
        self.s_space = s_space
        self.o_space = o_space
        self.t_model = t_model
        self.o_model = o_model
        self.r_model = r_model
        self.discount = discount

    # Action Space
    # State Space
    # Observation Space
    # Transition Model
    # Emission Model
    # Reward Model

    @abstractmethod
    def simulate(self, t_grid: torch.Tensor, agent, initial_state):
        raise NotImplementedError
    # Return trajectory




class DiscretePOMDP(POMDP):
    t_model: DiscreteTransitionModel
    o_model: PoissonObservationModel

    def __init__(self, a_space: Space, s_space: DiscreteSpace, o_space: Space, t_model: DiscreteTransitionModel,
                 o_model: PoissonObservationModel, r_model: RewardModel, discount: float):
        super().__init__(a_space, s_space, o_space, t_model, o_model, r_model, discount)

    def simulate(self, t_grid: torch.Tensor, agent: Agent, initial_state: torch.Tensor):
        t = t_grid[0].clone()
        T = t_grid[-1].clone()

        t_state_vector = torch.tensor([t]).clone()
        state_vector = initial_state[None, ...].clone()

        t_obs_vector = torch.tensor([])
        obs_vector = torch.tensor([])

        t_action_vector = torch.tensor([t]).clone()
        action_vector = agent.get_action(t)[None, ...].clone()

        t_reward_vector = torch.tensor([t]).clone()
        reward_vector = self.r_model(state_vector[-1], action_vector[-1])[None, ...].clone()

        while t_state_vector[-1] <= T:
            max_exit_rate = self.t_model.max_exit_rate(state_vector[-1])
            if max_exit_rate == torch.tensor(0.):
                min_waiting = T
            else:
                min_waiting = torch.distributions.exponential.Exponential(max_exit_rate).sample()

            t_obs_new = self.o_model.sample_times([t, (t + min_waiting).clip(max=T)])
            t_grid_new = t_grid[torch.logical_and(t < t_grid, t_grid <= (t + min_waiting))]

            t_event_new, srt_idx = torch.sort(torch.cat((t_obs_new, t_grid_new)))
            t_event_is_obs = torch.cat((torch.ones_like(t_obs_new), torch.zeros_like(t_grid_new)))[srt_idx]

            for idx, t_event in enumerate(t_event_new):
                # sample new observation
                action = agent.get_action(t_event)
                t_action_vector = torch.cat((t_action_vector, t_event.clone()[None, ...]))
                action_vector = torch.cat((action_vector, action.clone()[None, ...]))

                t_reward_vector = torch.cat((t_reward_vector, t_event.clone()[None, ...]))
                reward_vector = torch.cat(
                    (reward_vector, self.r_model(state_vector[-1], action_vector[-1]).clone()[None, ...]))

                if t_event_is_obs[idx]:
                    obs = self.o_model.sample(state_vector[-1], action)
                    # update agent
                    agent.add_observation(obs, t_event)

                    t_obs_vector = torch.cat((t_obs_vector, t_event.clone()[None, ...]))
                    obs_vector = torch.cat((obs_vector, obs.clone()[None, ...]))

                else:
                    t_state_vector = torch.cat((t_state_vector, t_event.clone()[None, ...]))
                    state_vector = torch.cat((state_vector, state_vector[-1].clone()[None, ...]))

            t += min_waiting
            if t > T:
                break

            #   waiting_time, new_state, observations, rewards = self._simulate_transition(t_start=t, state_start=state,agent)
            # Thinning
            action = agent.get_action(t)
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
        obs_trajectory = DiscreteTimeTrajectory(t_obs_vector, obs_vector)
        action_trajectory = ContinuousTimeTrajectory(t_action_vector, action_vector, interp_kind='next')
        reward_trajectory = ContinuousTimeTrajectory(t_reward_vector, reward_vector, interp_kind='next')

        return state_trajectory, obs_trajectory, action_trajectory, reward_trajectory

    def create_MDP(self):
        mdp = DiscreteMDP(a_space=self.a_space, s_space=self.s_space, t_model=self.t_model, r_model=self.r_model,
                  discount=self.discount)
        return mdp

class ContinuousPOMDP(POMDP):
    o_model: PoissonObservationModel
    t_model: ContinuousTransitionModel

    def __init__(self, a_space: Space, s_space: ContinuousSpace, o_space: Space, t_model: ContinuousTransitionModel,
                 o_model: PoissonObservationModel, r_model: RewardModel, discount: float):
        super().__init__(a_space, s_space, o_space, t_model, o_model, r_model, discount)

    def simulate(self, t_grid: torch.Tensor, agent, initial_state):

        ##TODO add reward vector
        t = t_grid[0]
        T = t_grid[-1]

        t_state_vector = torch.tensor([t]).clone()
        state_vector = initial_state[None, ...].clone()

        t_obs_vector = self.o_model.sample_times([t, T])
        obs_vector = torch.tensor([])

        # t_action_vector = torch.tensor([t]).clone()
        action_vector = agent.get_action(t)[None, ...].clone()

        t_reward_vector = torch.tensor([t]).clone()
        reward_vector = self.r_model(state_vector[-1], action_vector[-1])[None, ...].clone()

        for time in t_obs_vector:
            t_vector_new, state_vector_new, action_vector_new = self.solve_euler_maruyama(state_vector[-1],
                                                                                          t_state_vector[-1], time,
                                                                                          agent)
            t_state_vector = torch.cat((t_state_vector[:-1], t_vector_new))
            state_vector = torch.cat((state_vector[:-1], state_vector_new))
            action_vector = torch.cat((action_vector[:-1], action_vector_new))
            observation = self.o_model.sample(state_vector[-1], action_vector[-1])
            agent.add_observation(observation, time)
            obs_vector = torch.cat((obs_vector, observation[None]))
        # one last simulation till end point
        t_vector_new, state_vector_new, action_vector_new = self.solve_euler_maruyama(state_vector[-1],
                                                                                      t_state_vector[-1], T, agent)
        t_state_vector = torch.cat((t_state_vector[:-1], t_vector_new))
        state_vector = torch.cat((state_vector[:-1], state_vector_new))
        action_vector = torch.cat((action_vector[:-1], action_vector_new))

        state_trajectory = ContinuousTimeTrajectory(t_state_vector, state_vector)
        obs_trajectory = DiscreteTimeTrajectory(t_obs_vector, obs_vector)
        action_trajectory = ContinuousTimeTrajectory(t_state_vector, action_vector, interp_kind='next')
        reward_trajectory = ContinuousTimeTrajectory(t_reward_vector, reward_vector,interp_kind='next')

        #return t_state_vector, state_vector, action_vector, t_obs_vector, obs_vector
        return state_trajectory, obs_trajectory, action_trajectory, reward_trajectory


    def solve_euler_maruyama(self, x_init: torch.Tensor, t_init, t_end, agent, stepsize=torch.tensor(0.001)):
        """
        Solve sde with euler maruyama discetization in between t_init and t_end
        :param x_init: initial state
        :param t_init: initial time
        :param t_end: stopping time
        :param stepsize: stepsize for discretization
        :param agent: agent, that chooses action
        :return: trajectory of state and action
        """
        ### TODO add reward vector?
        state_vector = x_init[None, ...].clone()
        t_vector = torch.tensor([t_init]).clone()
        action = agent.get_action(t_vector[-1])
        action_vector = action[None, ...].clone()
        while t_vector[-1] + stepsize < t_end:
            x_ = state_vector[-1] + self.t_model.drift(state_vector[-1],
                                                       action_vector[-1]) * stepsize + self.t_model.dispersion(
                state_vector[-1], action_vector[-1]) @ torch.normal(torch.tensor([0.0, 0.0]), std=torch.sqrt(stepsize))
            t = t_vector[-1] + stepsize
            state_vector = torch.cat((state_vector, x_.clone()[None]))
            t_vector = torch.cat((t_vector, t.clone()[None]))
            action_vector = torch.cat((action_vector, agent.get_action(t).clone()[None]))

        # Final Step (smaller than the previous)
        stepsize = t_end - t_vector[-1]
        x_ = state_vector[-1] + self.t_model.drift(state_vector[-1],
                                                   action_vector[-1]) * stepsize + self.t_model.dispersion(
            state_vector[-1], action_vector[-1]) @ torch.normal(torch.tensor([0.0, 0.0]), std=torch.sqrt(stepsize))
        t = t_vector[-1] + stepsize
        state_vector = torch.cat((state_vector, x_.clone()[None]))
        t_vector = torch.cat((t_vector, t.clone()[None]))
        action_vector = torch.cat((action_vector, agent.get_action(t).clone()[None]))

        return t_vector, state_vector, action_vector




class DiscreteStateContTimePOMDP(POMDP):
    #Observation Model is in continuous time
    t_model: DiscreteTransitionModel
    o_model: ExactContTimeObservationModel

    def __init__(self, a_space: Space, s_space: DiscreteSpace, o_space: Space, t_model: DiscreteTransitionModel,
                 o_model: ExactContTimeObservationModel, r_model: RewardModel, discount: float):
        super().__init__(a_space, s_space, o_space, t_model, o_model, r_model, discount)

    def simulate(self, t_grid: torch.Tensor, agent: Agent, initial_state: torch.Tensor):
        t = t_grid[0].clone()
        T = t_grid[-1].clone()

        t_state_vector = torch.tensor([t]).clone()
        state_vector = initial_state[None, ...].clone()

        #t_obs_vector = torch.tensor([])
        #obs_vector = torch.tensor([])

        t_action_vector = torch.tensor([t]).clone()
        action_vector = agent.get_action(t)[None, ...].clone()

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
                action = agent.get_action(t_event)
                t_action_vector = torch.cat((t_action_vector, t_event.clone()[None, ...]))
                action_vector = torch.cat((action_vector, action.clone()[None, ...]))

                t_reward_vector = torch.cat((t_reward_vector, t_event.clone()[None, ...]))
                reward_vector = torch.cat(
                    (reward_vector, self.r_model(state_vector[-1], action_vector[-1]).clone()[None, ...]))
                #
                # if t_event_is_obs[idx]:
                #     obs = self.o_model.sample(state_vector[-1], action)
                #     # update agent
                #     agent.add_observation(obs, t_event)
                #
                #     t_obs_vector = torch.cat((t_obs_vector, t_event.clone()[None, ...]))
                #     obs_vector = torch.cat((obs_vector, obs.clone()[None, ...]))

                #else:
                t_state_vector = torch.cat((t_state_vector, t_event.clone()[None, ...]))
                state_vector = torch.cat((state_vector, state_vector[-1].clone()[None, ...]))

            t += min_waiting
            if t > T:
                break

            #   waiting_time, new_state, observations, rewards = self._simulate_transition(t_start=t, state_start=state,agent)
            # Thinning
            action = agent.get_action(t)
            t_action_vector = torch.cat((t_action_vector, t.clone()[None, ...]))
            action_vector = torch.cat((action_vector, action.clone()[None, ...]))

            t_reward_vector = torch.cat((t_reward_vector, t.clone()[None, ...]))
            reward_vector = torch.cat(
                (reward_vector, self.r_model(state_vector[-1], action_vector[-1]).clone()[None, ...]))

            if torch.rand(1) <= self.t_model.exit_rate(state_vector[-1], action) / max_exit_rate:
                next_state = self.t_model.sample_next_state(state_vector[-1], action)
                t_state_vector = torch.cat((t_state_vector, t.clone()[None, ...]))
                state_vector = torch.cat((state_vector, next_state.clone()[None, ...]))
                #Update agent
                #maybe not sample?
                new_obs = self.o_model(state_vector[-1])
                agent.add_observation(new_obs,t)

        state_trajectory = ContinuousTimeTrajectory(t_state_vector, state_vector)
        #obs_trajectory = DiscreteTimeTrajectory(t_obs_vector, obs_vector)
        action_trajectory = ContinuousTimeTrajectory(t_action_vector, action_vector, interp_kind='nearest')
        reward_trajectory = ContinuousTimeTrajectory(t_reward_vector, reward_vector)

        return state_trajectory, action_trajectory, reward_trajectory


    def create_MDP(self):
        mdp = DiscreteMDP(a_space=self.a_space, s_space=self.s_space, t_model=self.t_model, r_model=self.r_model,
                  discount=self.discount)
        return mdp

