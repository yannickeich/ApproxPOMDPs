import matplotlib.pyplot as plt
import torch
import numpy as np
from packages.model.components.agents.discrete_time_obs.discrete_spaces.discrete_parametric_agent import BinomialQueueingAgent
from packages.model.examples.Queueing.Queueing_network import QueueingPOMDP
from _examples.figure_configuration_aistats import figure_configuration_aistats

# Create Plot with particle and projection filter based on results from Particle.py and BinomialQueuing_Qfunction.py

#Create Problem and Agent
pomdp = QueueingPOMDP()
agent = BinomialQueueingAgent(pomdp.t_model,pomdp.o_model,initial_param=torch.tensor([0.5,0.5,0.5]),initial_time=torch.tensor(0.))


#Load data here
#From BinomialQueuing_Qfunction.py
state_trajectory = torch.load('state_traj.pt')
obs_trajectory = torch.load('obs_traj.pt')
action_trajectory = torch.load('action_traj.pt')
filter_trajectory = torch.load('filter_traj.pt')

#From Particle.py
particle_state_trajectory = torch.load('state_trajs.pt')


num_particles = len(particle_state_trajectory)

t_grid = torch.tensor([0.0,10.0])

### PLOTS
t = torch.linspace(0,t_grid[-1],1000)

projection_mean, projection_var, _ = agent.moments(torch.tensor(filter_trajectory(t)))
particle_trajectories = torch.zeros(num_particles,1000,3)
for j in range(num_particles):
    particle_trajectories[j] = torch.tensor(particle_state_trajectory[j]['trajectory'](t))
mean = particle_trajectories.mean(dim=0)
std = particle_trajectories.var(dim=0)

figure_configuration_aistats(k_width_height=1.03)
fig, axs = plt.subplots(4,1,sharex=True)

for i in range(3):
    axs[i].plot(t,state_trajectory(t)[:,i],'black')
    axs[i].plot(t,projection_mean[:,i],color='b',linestyle='--')
    axs[i].fill_between(t,projection_mean[:,i]+np.sqrt(projection_var[:,i]),projection_mean[:,i]-np.sqrt(projection_var[:,i]),color='b',alpha=0.2)
    axs[i].plot(t,mean[:,i],'red',ls='-.')
    axs[i].fill_between(t,mean[:,i]+np.sqrt(std[:,i]),mean[:,i]-np.sqrt(std[:,i]),color='r',alpha=0.2)

state1=axs[0].plot(t, state_trajectory(t)[:, 0], 'k')
axs[1].plot(obs_trajectory.times, obs_trajectory.values[..., 0], 'x', color='green')
axs[2].plot(obs_trajectory.times, obs_trajectory.values[..., 1], 'x', color='green')

axs[0].set_ylabel(r'$x_1(t)$')
axs[1].set_ylabel(r'$x_2(t)$')
axs[2].set_ylabel(r'$x_3(t)$')
axs[-1].plot(t, action_trajectory(t),'purple',linewidth=0.5)
axs[-1].set(ylabel=r'u(t)')
plt.xlabel('Time ' + r"$t$" + " in " + r"$s$")

fig.tight_layout(pad=0.0)
plt.show()

