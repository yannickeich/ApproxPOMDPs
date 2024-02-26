import torch
from packages.model.examples.MonoCRN import MultinomialMonoCRNPOMDP
from packages.solvers.sampler import MultinomialSampler
from packages.solvers.MDP_Q_Learning import Q_Learning_Solver


# Solve the chemical reaction network MDP using Q Learning


#Create Problem
## 4 states, 2 estimates, 2 observed
total_count = 300
state_dim = 4
obs_dim = 2
c = torch.zeros(2,4,4)
c[0]= torch.tensor([[0.,1.,1.,0.],[0.,0.,0.,1.],[1.,0.,0.,1.],[0.,1.,1.,0.]])
c[1]= torch.tensor([[0.,0.,1.,0.],[1.,0.,0.,1.],[1.,0.,0.,1.],[0.,1.,1.,0.]])
c = c/20
pomdp = MultinomialMonoCRNPOMDP(c, obs_dim=obs_dim, total_N=total_count)
mdp = pomdp.create_MDP()


# Solve MDP using Q_Learning
normalization = total_count/4
sampler = MultinomialSampler(total_count=total_count,state_dim=state_dim)
solver = Q_Learning_Solver(mdp=mdp,sampler=sampler,normalization=normalization,iterations=400000)
solver.solve()