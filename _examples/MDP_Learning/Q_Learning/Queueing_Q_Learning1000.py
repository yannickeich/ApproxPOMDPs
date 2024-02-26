import torch
from packages.model.examples.Queueing.Queueing_network import QueueingMDP
from packages.solvers.MDP_Q_Learning import Q_Learning_Solver
from packages.solvers.sampler import DiscreteUniformSampler

# Solve Queueing MDP using Q Learning

buffer_sizes = torch.tensor([1000,1000,1000])
mdp = QueueingMDP(buffer_sizes=buffer_sizes)

bounds = torch.tensor([[0,1001],[0,1001],[0,1001]])
sampler = DiscreteUniformSampler(bounds=bounds)


solver = Q_Learning_Solver(mdp=mdp,sampler=sampler,normalization=1000,iterations = 400000)
q_value_net = solver.solve()
