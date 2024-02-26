import torch

from packages.model.examples.Queueing.Queueing_network import QueueingPOMDP
from packages.solvers.particle_filter import ParticleFilter

# ParticleFilter for Comparison, given a trajectory

# Create Problem
pomdp = QueueingPOMDP()
mdp = pomdp.create_MDP()
o_model = pomdp.o_model

# Load sample trajectory
state_trajectory = torch.load('state_traj.pt')
obs_trajectory = torch.load('obs_traj.pt')
action_trajectory = torch.load('action_traj.pt')


t_grid = torch.tensor([0.0,10.0])

initial_state = torch.tensor([800,700,500])
### Particle_filter

num_particles = 10000
solver = ParticleFilter(mdp,o_model,action_trajectory,obs_trajectory,t_start = t_grid[0], t_final = t_grid[-1], num_particles = num_particles, initial_state = initial_state)
state_trajs = solver.solve()
torch.save(state_trajs,'state_trajs.pt')

print('debug')