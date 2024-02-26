from packages.model.mdp import DiscreteMDP
import torch
import numpy as np

class FixedPointIteration:
    def  __init__(self,DiscreteMDP,iterations = 5000):
        self.DiscreteMDP = DiscreteMDP
        self.iterations = iterations
    def solve(self):
        ### Tabular Method to solve MDP

        states = self.DiscreteMDP.s_space.elements
        actions = self.DiscreteMDP.a_space.elements

        # Initialize V
        V = torch.zeros(states.shape[0])

        rates, next_states = self.DiscreteMDP.t_model.rates(states[None, ...], actions[:, None, :])

        exit_rates = rates.sum(0)
        try:
            index = np.ravel_multi_index(next_states.flatten(end_dim=-2).numpy().T, self.DiscreteMDP.s_space.cardinalities)
        except:
            index = (next_states.reshape(-1,states.shape[-1])[...,None,:]==states).prod(-1).nonzero()[...,-1]
        for i in range(self.iterations):
            V_old = V.clone()
            V,_ = ((self.DiscreteMDP.r_model(states[None, ...], actions[:, None, ...])/self.DiscreteMDP.discount + (
                        rates * V[index].reshape(*next_states.shape[:-1])).sum(0)) / (
                            1 / self.DiscreteMDP.discount + exit_rates)).max(dim=0)

            if i%100==0:
                objective = ((V-V_old)**2).sum()
                print(objective)

        Q = self.DiscreteMDP.r_model(states[None, ...], actions[:, None, ...]) + self.DiscreteMDP.discount * (
                rates * V[index].reshape(*next_states.shape[:-1])).sum(0) - self.DiscreteMDP.discount * rates.sum(0) * V
        return Q