import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from packages.model.examples.Queueing.Queueing_network import QueueingPOMDP
from packages.model.components.agents.agent import HistogramAgent
from packages.model.components.agents.discrete_time_obs.discrete_spaces.discrete_parametric_agent import BinomialAgent
from packages.solvers.mdp_contraction_solver import FixedPointIteration
from packages.model.mdp import MDPAgent
from _examples.figure_configuration_aistats import figure_configuration_aistats


## Compare Exact Filter with QMDP vs Projection Filter with MDP

# Create Problem
buffer_sizes = torch.tensor([5,5,5])
birth = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
death = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
inter_rates = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                          [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]])
pomdp = QueueingPOMDP(buffer_sizes=buffer_sizes, birth = birth, death = death, inter_rates= inter_rates,scale=1,variance = 0.5*torch.eye(2))
mdp = pomdp.create_MDP()


# Create rates Table for Histogram agent
states = pomdp.s_space.elements
actions = pomdp.a_space.elements
n_states = states.shape[0]
n_actions = actions.shape[0]
rates, next_states = pomdp.t_model.rates(states[None, ...], actions[..., None, :])
# include "rate" to itself
rates = torch.cat((rates, -rates.sum(0)[None]), dim=0)
next_states = torch.cat((next_states, states.expand(*next_states.shape[1:])[None]), dim=0)

rates = rates.transpose(0, 1)
next_states = next_states.transpose(0, 1)

# index of the next_states
index = np.ravel_multi_index(next_states.flatten(end_dim=-2).numpy().T, pomdp.s_space.cardinalities)
index = torch.tensor(index.reshape(*rates.shape))

# Fill up the rates/transitions matrix
rates_matrix = torch.zeros((pomdp.a_space.elements.shape[0], pomdp.s_space.elements.shape[0],
                            pomdp.s_space.elements.shape[0]))
# rates_matrix[i][index[i][j][k]][k] = rates[i][j][k], j is the position of the next_state, thats why dim=-2, look for its index and place it at the new position - then transpose
rates_matrix = rates_matrix.scatter_add(dim=-2, index=index, src=rates).transpose(-1, -2)
pomdp.t_model.table = rates_matrix.permute(1,2,0)




# Simulation parameters
t_grid = torch.tensor([0.,10.])
t = torch.linspace(t_grid[0],t_grid[-1],1000)

initial_state = torch.tensor([1,1,1])
index = (mdp.s_space.elements ==  initial_state).prod(-1).argmax()
initial_hist = torch.zeros(mdp.s_space.elements.shape[0])
initial_hist[index] = 1.0
initial_param = torch.tensor([0.2,0.2,0.2])

#Solve MDP
solver = FixedPointIteration(mdp)
Q = solver.solve()


#MDP Control, exact knowledge
mdp_agent = MDPAgent(mdp.s_space,mdp.a_space,Q=Q)
mdp_samples = 100
cumulative_reward = torch.zeros(mdp_samples)

for i in range(mdp_samples):
    state_trajectory, action_trajectory, reward_trajectory = mdp.simulate(t_grid,agent=mdp_agent,initial_state=initial_state)
    cumulative_reward[i] = torch.tensor(reward_trajectory(t).mean())


# QMDP Method_ Binomial
pomdp_samples = 100
cumulative_reward2 = torch.zeros(pomdp_samples)
for i in range(pomdp_samples):
    ## Needs to be defined in the for loop, as it saves the belief
    binom_agent = BinomialAgent(pomdp.t_model,pomdp.o_model,initial_param = initial_param,initial_time=t_grid[0],Q_matrix=Q,exp_method='exact')
    state_trajectory, obs_trajectory, action_trajectory, reward_trajectory = pomdp.simulate(t_grid,agent=binom_agent, initial_state = initial_state)
    cumulative_reward2[i] = torch.tensor(reward_trajectory(t).mean())
    print(i)


#QMDP Method_ Exact
pomdp_samples = 100
cumulative_reward3 = torch.zeros(pomdp_samples)


for i in range(pomdp_samples):
    ## Needs to be defined in the for loop, as it saves the belief
    exact_agent = HistogramAgent(pomdp.t_model,pomdp.o_model,initial_hist=initial_hist,initial_time=t_grid[0],Q_matrix=Q)
    state_trajectory, obs_trajectory, action_trajectory, reward_trajectory = pomdp.simulate(t_grid,agent=exact_agent, initial_state = initial_state)
    cumulative_reward3[i] = torch.tensor(reward_trajectory(t).mean())
    print(i)




print(cumulative_reward.mean())
print(cumulative_reward2.mean())
print(cumulative_reward3.mean())



figure_configuration_aistats()
fig, ax = plt.subplots()
sns.kdeplot(cumulative_reward,ax=ax, color='black',fill=True,label='MDP Control')
sns.kdeplot(cumulative_reward2,ax=ax,color='blue',fill=True, label ='Projection Filter')
sns.kdeplot(cumulative_reward3,ax=ax,color='red',fill=True,label = 'Exact Filter')

ax.set_xlabel('Reward')
ax.set_ylabel('Density')
fig.tight_layout(pad=0)
print('debug')