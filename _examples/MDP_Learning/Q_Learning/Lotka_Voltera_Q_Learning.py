import torch
import matplotlib.pyplot as plt
import matplotlib

from packages.model.examples.traditional_lotka_volterra_reaction_network import LotkaVolterraPOMDP

from packages.solvers.MDP_Q_Learning import Q_Learning_Solver
from packages.solvers.sampler import DiscreteUniformSampler

# Learn LV mdp using Q_learning

#Create MDP and sample area
pomdp = LotkaVolterraPOMDP(scale = 20,goal=torch.tensor([100,100]))
mdp = pomdp.create_MDP()
bounds = torch.tensor([[0,250],[0,250]])
sampler = DiscreteUniformSampler(bounds=bounds)


solver = Q_Learning_Solver(mdp=mdp,sampler=sampler,normalization=250,iterations = 400000)
q_value_net = solver.solve()


