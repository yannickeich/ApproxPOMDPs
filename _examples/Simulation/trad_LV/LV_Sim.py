import torch
import numpy as np
from packages.model.examples.traditional_lotka_volterra_reaction_network import LotkaVolterraPOMDP
import matplotlib.pyplot as plt
from _examples.figure_configuration_aistats import figure_configuration_aistats
from packages.model.components.agents.discrete_time_obs.discrete_spaces.discrete_parametric_agent import PoissonCRNAgent
from packages.model.components.trajectories import ContinuousTimeTrajectory

# Create samples trajectories of LV problem with constant control

#Create Problem
pomdp = LotkaVolterraPOMDP(obs_rate=torch.tensor(1.0))



#Simulation
t_grid = torch.tensor([0.0,5.0])

initial_state = torch.tensor([50,50])
initial_param = initial_state.float()

agent = PoissonCRNAgent(pomdp.t_model, pomdp.o_model, initial_param, t_grid[0])

state_trajectory, obs_trajectory, action_trajectory, reward_trajectory = pomdp.simulate(t_grid,agent,initial_state)

filter_trajectory = ContinuousTimeTrajectory(agent.time_vector, agent.belief_vector, interp_kind='linear')
complete_action_trajectory = ContinuousTimeTrajectory(agent.action_time_vector, agent.action_vector)

### Plots
t = torch.linspace(0, t_grid[-1], 1000)
mean, var, _ = agent.moments(torch.tensor(filter_trajectory(t)))
figure_configuration_aistats(k_width_height=1.4)
fig, axs = plt.subplots(3, 1,sharex=True)


for i in range(2):
    axs[i].plot(t, state_trajectory(t)[:, i], 'k', label='state ' + str(i + 1))
    axs[i].plot(t, mean[:, i], 'b', label='belief ' + str(i + 1))

    axs[i].plot(obs_trajectory.times, obs_trajectory.values[..., i], 'x', label='observation ' + str(i + 1), color='green',markersize=6)
    axs[i].fill_between(t, mean[:, i] + np.sqrt(var[:, i]), mean[:, i] - np.sqrt(var[:, i]), color='b', alpha=0.2)

axs[2].plot(t, complete_action_trajectory(t),'purple',linewidth=0.5)
axs[0].set(ylabel=r'$x_1(t)$')
axs[1].set(ylabel=r'$x_2(t)$')

axs[2].set(ylabel=r'$u(t)$')


plt.xlabel('Time ' + r"$t$" + " in " + r"$s$")

fig.tight_layout(pad=0.0)
plt.show()

