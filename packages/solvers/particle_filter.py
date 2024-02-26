from packages.model.mdp import MDPAgent
import torch
from packages.model.components.trajectories import ContinuousTimeTrajectory

class ParticleFilter():
    def __init__(self,mdp,o_model,action_trajectory,obs_trajectory,t_start,t_final,num_particles,initial_state):
        """

        :param mdp: MDP, where the particles are being sampled.
        :param o_model: The observation model, that produced the ground truth observations.
        :param action_trajectory: The action_trajectory, that all particles should follow.
        :param obs_trajectory: The observation trajectory, observed from the ground truth.
        :param t_grid: time grid for the simulation.
        :param num_particles: number of particles.
        :param initial_state: initial state for each particle
        """
        self.mdp = mdp
        self.o_model = o_model
        self.t_grid = torch.tensor([t_start,t_final])
        self.num_particles = num_particles
        self.initial_state = initial_state
        self.mdp_agent = MDPAgent(s_space=mdp.s_space,a_space=mdp.a_space,action_trajectory=action_trajectory)
        self.obs_trajectory = obs_trajectory

    def solve(self):
        obs_times = self.obs_trajectory.times
        obs_values = self.obs_trajectory.values

        t_grid_new = torch.concat((self.t_grid[0][None],obs_times,self.t_grid[1][None]))

        # Number of particles
        initial_states = self.initial_state.repeat(self.num_particles, 1)
        state_traj = [{"values": torch.tensor([]), 'times': torch.tensor([])} for i in range(self.num_particles)]


        for i in range(t_grid_new.shape[0] - 1):
            t_small_grid = torch.tensor([t_grid_new[i], t_grid_new[i + 1]])
            #t = torch.linspace(t_small_grid[0], t_small_grid[1], 100)
            for j in range(self.num_particles):
                state_trajectory, action_trajectory, reward_trajectory = self.mdp.simulate(t_small_grid, self.mdp_agent,
                                                                                         initial_state=initial_states[j])
                state_traj[j]['values'] = torch.cat((state_traj[j]['values'],state_trajectory.values),dim=0)
                state_traj[j]['times'] = torch.cat((state_traj[j]['times'], state_trajectory.times), dim=0)

            #Gather the last states of all sample paths in a tensor
            current_states_list = [state_dict['values'][-1][None] for state_dict in state_traj]
            current_states = torch.cat(current_states_list,dim=0)

            if i < t_grid_new.shape[0] - 2:
                log_likelihood = self.o_model.log_prob(current_states, action=torch.tensor([0]), observation=obs_values[i])
                samples = torch.distributions.categorical.Categorical(logits=log_likelihood).sample((self.num_particles,))
                initial_states = current_states[samples]


        for j in range(self.num_particles):
           state_traj[j]["trajectory"] = ContinuousTimeTrajectory(state_traj[j]['times'],state_traj[j]['values'])
        return state_traj