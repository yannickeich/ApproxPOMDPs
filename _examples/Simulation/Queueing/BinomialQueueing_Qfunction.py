import torch
import numpy as np
from packages.model.examples.Queueing.Queueing_network import QueueingMDP,QueueingPOMDP
import matplotlib.pyplot as plt
from packages.model.components.agents.discrete_time_obs.discrete_spaces.discrete_parametric_agent import BinomialQueueingAgent
from packages.solvers.MDP_Q_Learning import Q_Learning_Solver
from packages.solvers.sampler import DiscreteUniformSampler
from packages.model.components.trajectories import ContinuousTimeTrajectory, DiscreteTimeTrajectory
from packages.solvers.particle_filter import ParticleFilter
from _examples.figure_configuration_aistats import figure_configuration_aistats

# Entropic Matching + QMDP Control for the queueing problem based on solving the MDP with Q_Learning
#

# Create Problem
pomdp = QueueingPOMDP(obs_rate=torch.tensor(0.5))
mdp = pomdp.create_MDP()


# Q-Learning...
bounds = torch.tensor([[0,1001],[0,1001],[0,1001]])
normalization = 1000
sampler = DiscreteUniformSampler(bounds=bounds)
solver = Q_Learning_Solver(mdp=mdp,sampler=sampler,normalization=normalization,iterations=400000)
#L
#q_value_net = solver.solve()

#... or load learnt Q-function
q_value_net = torch.load('prelearnt_Q_function_Queueing.pt')



#Simulation
t_grid = torch.tensor([0.0,10.0])
initial_state = torch.tensor([800,700,500])
initial_param = initial_state/pomdp.s_space.buffer_sizes

agent = BinomialQueueingAgent(pomdp.t_model, pomdp.o_model, initial_param, t_grid[0], Q_function = q_value_net, normalization=normalization,exp_samples=10)

state_trajectory, obs_trajectory, action_trajectory, reward_trajectory = pomdp.simulate(t_grid,agent,initial_state)
filter_trajectory = ContinuousTimeTrajectory(agent.time_vector,agent.belief_vector,interp_kind='linear')
complete_action_trajectory = ContinuousTimeTrajectory(agent.action_time_vector,agent.action_vector)



### PLOTS
t = torch.linspace(0,t_grid[-1],1000)
mean, var, _ = agent.moments(torch.tensor(filter_trajectory(t)))
figure_configuration_aistats(k_width_height=0.75 * 1.3)
fig, axs = plt.subplots(4,1,sharex=True)

for i in range(3):
    axs[i].plot(t,state_trajectory(t)[:,i],'k')
    axs[i].plot(t,mean[:,i],'b')

    axs[i].fill_between(t,mean[:,i]+np.sqrt(var[:,i]),mean[:,i]-np.sqrt(var[:,i]),color='b',alpha=0.3)

state1=axs[0].plot(t, state_trajectory(t)[:, 0], 'k')
filter1=axs[0].plot(t, mean[:, 0], 'b', label='Filter')
axs[1].plot(obs_trajectory.times, obs_trajectory.values[..., 0], 'x', color='r', alpha=0.8)
axs[2].plot(obs_trajectory.times, obs_trajectory.values[..., 1], 'x', color='r', alpha=0.8)

axs[0].set_ylabel(r'$x_1(t)$')
axs[1].set_ylabel(r'$x_2(t)$')
axs[2].set_ylabel(r'$x_3(t)$')

axs[-1].plot(t, complete_action_trajectory(t),'purple')
axs[-1].set(ylabel=r'u(t)')
plt.xlabel('Time ' + r"$t$" + " in " + r"$s$")
fig.tight_layout(pad=0.0)
plt.show()
