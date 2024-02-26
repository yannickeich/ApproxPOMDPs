import torch
from packages.model.examples.MonoCRN import MultinomialMonoCRNPOMDP
from packages.model.components.agents.cont_time_obs.cont_time_agent import MultinomialMonoCRNAgent
from packages.model.components.trajectories import ContinuousTimeTrajectory
from packages.solvers.sampler import MultinomialSampler
from packages.solvers.MDP_Q_Learning import Q_Learning_Solver
import matplotlib.pyplot as plt
import numpy as np
from _examples.figure_configuration_aistats import figure_configuration_aistats

# Entropic Matching + QMDP Control for the chemical reaction network based on solving the MDP with Q_Learning


#Create Problem
## 4 states, 2 estimates, 2 observed
torch.manual_seed(1028)
total_count = 300
state_dim = 4
obs_dim = 2
c = torch.zeros(2,4,4)
c[0]= torch.tensor([[0.,1.,1.,0.],[0.,0.,0.,1.],[1.,0.,0.,1.],[0.,1.,1.,0.]])
c[1]= torch.tensor([[0.,0.,1.,0.],[1.,0.,0.,1.],[1.,0.,0.,1.],[0.,1.,1.,0.]])
c = c/20
pomdp = MultinomialMonoCRNPOMDP(c, obs_dim=obs_dim, total_N=total_count)
mdp = pomdp.create_MDP()




# Q Learning...
normalization = total_count/4
sampler = MultinomialSampler(total_count=total_count,state_dim=state_dim)
solver = Q_Learning_Solver(mdp=mdp,sampler=sampler,normalization=normalization,iterations=40000)
#q_value_net = solver.solve()

#... or load learnt approximation
q_value_net = torch.load('Q_function_Multinomial.pt')



# Create EntropicMatching Agent
agent = MultinomialMonoCRNAgent(pomdp.t_model, pomdp.o_model, Q_function=q_value_net,normalization=normalization, initial_param = torch.tensor([8/15, 100., 50.]), initial_time= torch.tensor(0.0))


# Simulate
t_grid = torch.linspace(0, 10, 1000)
state_trajectory, action_trajectory, reward_trajectory = pomdp.simulate(t_grid= t_grid,agent=agent,initial_state=torch.tensor([80,70,100,50]))
filter_trajectory = ContinuousTimeTrajectory(agent.time_vector,agent.belief_vector,interp_kind='linear')
complete_action_trajectory = ContinuousTimeTrajectory(agent.action_time_vector,agent.action_vector)



### Plots
t = torch.linspace(0, t_grid[-1], 1000)
mean, var, _ = agent.moments(torch.tensor(filter_trajectory(t)))

figure_configuration_aistats(k_width_height=1.03)
fig, axs = plt.subplots(4,1,sharex=True)

axs[0].plot(t, state_trajectory(t)[:, 0], 'black')
axs[1].plot(t, state_trajectory(t)[:, 1], 'black')
axs[2].plot(t, state_trajectory(t)[:, 2], 'red')
axs[2].plot(t, state_trajectory(t)[:, 3], 'green')


axs[0].plot(t,mean[:, 0], 'blue')
axs[1].plot(t, mean[:, 1], 'blue')
axs[0].fill_between(t, mean[:, 0] + np.sqrt(var[:, 0]), mean[:, 0] - np.sqrt(var[:, 0]),color='b', alpha=0.2)
axs[1].fill_between(t, mean[:, 1] + np.sqrt(var[:, 1]), mean[:, 1] - np.sqrt(var[:, 1]),color='b', alpha=0.2)

axs[0].set_ylabel(r'$x_1(t)$')
axs[1].set_ylabel(r'$x_2(t)$')
axs[2].set_ylabel(r'$\bar{x}(t)$')

axs[-1].plot(t, complete_action_trajectory(t),'purple',linewidth=0.5)
axs[-1].set_ylabel(r'$u(t)$')
axs[-1].set_xlabel('Time ' + r"$t$" + " in " + r"$s$")
fig.tight_layout(pad=0.0)
plt.show()

