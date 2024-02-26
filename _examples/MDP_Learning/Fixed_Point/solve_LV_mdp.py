import torch
import matplotlib.pyplot as plt
import matplotlib
from packages.model.components.agents.discrete_time_obs.discrete_spaces.discrete_parametric_agent import PoissonAgent
# from packages.model.examples.lotka_volterra_reaction_network import LotkaVolterraMDP,LotkaVolterraPOMDP
from packages.model.examples.traditional_lotka_volterra_reaction_network import LotkaVolterraMDP, LotkaVolterraPOMDP
from packages.solvers.mdp_contraction_solver import FixedPointIteration
from packages.model.components.transition_model import TruncCRNTransitionModel
from packages.model.components.spaces import FiniteDiscreteSpace


#Solve LV Model via Fixed Point Iteration
# State Space needs to be truncated
# Change with a finite transition_model to use truncated tabular version,


pomdp = LotkaVolterraPOMDP()
mdp = pomdp.create_MDP()
truncation = [500,500]
state_space = FiniteDiscreteSpace(truncation)
transition_model = TruncCRNTransitionModel(state_space,mdp.a_space,mdp.t_model.S,mdp.t_model.P,mdp.t_model.c)
mdp.t_model=transition_model
mdp.s_space=state_space

solver = FixedPointIteration(mdp)
Q = solver.solve()

