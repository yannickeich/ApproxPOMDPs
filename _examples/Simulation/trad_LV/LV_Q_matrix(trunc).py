import matplotlib.pyplot as plt
from _examples.figure_configuration_aistats import figure_configuration_aistats
import torch
import numpy as np
from packages.model.examples.traditional_lotka_volterra_reaction_network import LotkaVolterraPOMDP

from packages.model.components.agents.discrete_time_obs.discrete_spaces.discrete_parametric_agent import PoissonCRNAgent
from packages.solvers.mdp_contraction_solver import FixedPointIteration
from packages.model.components.trajectories import ContinuousTimeTrajectory
from packages.model.components.spaces import FiniteDiscreteSpace
from packages.model.components.transition_model import TruncCRNTransitionModel


# LV Projection + QMDP based on FixedPointIteration
# Truncate State space, so FixedPointIteration is tractable



#Create Problem
pomdp = LotkaVolterraPOMDP(scale = 20,goal=torch.tensor([100,100]))
mdp = pomdp.create_MDP()
truncation = [1000,1000]
state_space = FiniteDiscreteSpace(truncation)
transition_model = TruncCRNTransitionModel(state_space,mdp.a_space,mdp.t_model.S,mdp.t_model.P,mdp.t_model.c)
mdp.t_model=transition_model
mdp.s_space=state_space
pomdp.t_model = transition_model
mdp.s_space = state_space

#Solve MDP
solver = FixedPointIteration(mdp,iterations=80000)
Q = solver.solve()

#Simulation
t_grid = torch.tensor([0.0,5.0])
initial_state = torch.tensor([50,50])
initial_param = initial_state.float()
agent = PoissonCRNAgent(pomdp.t_model, pomdp.o_model, initial_param, t_grid[0], Q_matrix=Q)
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

    axs[i].plot(obs_trajectory.times, obs_trajectory.values[..., i], 'x', label='observation ' + str(i + 1), color='r')
    axs[i].fill_between(t, mean[:, i] + np.sqrt(var[:, i]), mean[:, i] - np.sqrt(var[:, i]), color='b', alpha=0.2)

axs[2].plot(t, complete_action_trajectory(t),'purple',linewidth=0.5)
axs[0].set(ylabel=r'$x_1(t)$')
axs[1].set(ylabel=r'$x_2(t)$')
axs[2].set(ylabel=r'$u(t)$')

plt.xlabel('Time ' + r"$t$" + " in " + r"$s$")
fig.tight_layout(pad=0.0)
plt.show()

