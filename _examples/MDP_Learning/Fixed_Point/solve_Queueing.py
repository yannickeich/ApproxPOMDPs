import torch
from packages.model.examples.Queueing.Queueing_network import QueueingMDP
from packages.solvers.mdp_contraction_solver import FixedPointIteration
from packages.solvers.sampler import DiscreteUniformSampler


## Show FixedPointIteration for a small MDP

### Queueing System
### State 3 Dim //  2 Actions
buffer_sizes = torch.tensor([10,10,10])
obs_rate=torch.tensor(0.0)
birth = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
death = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
rates = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
mdp = QueueingMDP(buffer_sizes=buffer_sizes, birth = birth, inter_rates= rates, death = death,scale=10)
solver = FixedPointIteration(mdp)
Q = solver.solve()

normalization = 10
bounds = torch.tensor([[0,11],[0,11],[0,11]])
sampler = DiscreteUniformSampler(bounds)
