import torch
import matplotlib.pyplot as plt
from packages.model.components.agents.cont_time_obs.cont_time_agent import MultinomialMonoCRNAgent
from packages.model.examples.MonoCRN import MultinomialMonoCRNMDP,MultinomialMonoCRNPOMDP
from packages.solvers.mdp_contraction_solver import FixedPointIteration
from _examples.figure_configuration_aistats import figure_configuration_aistats


#Solve small CRN with exact subsystem measurement using Fixed Point Iteration

state_dim = 4
obs_dim = 2
total_N=12
c = torch.zeros(2,4,4)
c[0]= torch.tensor([[0.,1.,1.,0.],[0.,0.,0.,1.],[1.,0.,0.,1.],[0.,1.,1.,0.]])
c[1]= torch.tensor([[0.,0.,1.,0.],[1.,0.,0.,1.],[1.,0.,0.,1.],[0.,1.,1.,0.]])
c = c/5
mdp = MultinomialMonoCRNMDP(c=c,total_N=12)



solver = FixedPointIteration(mdp)
Q = solver.solve()
