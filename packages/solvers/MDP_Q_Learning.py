
from packages.solvers.sampler import Sampler
from packages.model.mdp import DiscreteMDP
import torch
import torch.nn as nn

class Q_Learning_Solver:
    def  __init__(self,mdp:DiscreteMDP, sampler: Sampler ,q_value_net = None , opt_constructor = None, optimizer_options = None, normalization = None, scheduler = None,iterations = 10000):
        self.mdp = mdp
        self.sampler = sampler
        self.iterations = iterations

        if q_value_net is None:
            q_value_net = nn.Sequential(nn.Linear(in_features= mdp.s_space.dimensions ,out_features=200,bias=True),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features=200, out_features=200, bias=True),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features=200,out_features=200,bias=True),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features = 200,out_features = mdp.a_space.elements.size(0)))

            q_target_net  = nn.Sequential(nn.Linear(in_features= mdp.s_space.dimensions ,out_features=200,bias=True),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features=200, out_features=200, bias=True),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features=200,out_features=200,bias=True),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features = 200,out_features = mdp.a_space.elements.size(0)))
        else:
            q_target_net = q_value_net
        self.q_value_net = q_value_net
        self.q_target_net = q_target_net
        target_net_state_dict = self.q_target_net.state_dict()
        value_net_state_dict = self.q_value_net.state_dict()
        for key in value_net_state_dict:
            target_net_state_dict[key] = value_net_state_dict[key] * 0.999 + target_net_state_dict[key] * (1 - 0.999)
        self.q_target_net.load_state_dict(target_net_state_dict)
        if opt_constructor is None:
            opt_constructor= torch.optim.Adam
        if optimizer_options is None:
            optimizer_options = dict()
        self.opt_constructor = opt_constructor
        self.optimizer_v_options = optimizer_options

        self.optimizer = opt_constructor(self.q_value_net.parameters(),**self.optimizer_v_options)
        if normalization is None:
            normalization = 1
        self.normalization = normalization

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,step_size=20000,gamma=0.2)
        self.scheduler = scheduler

    def solve(self):
        batch_size = 64
        TAU = 0.001
        actions = self.mdp.a_space.elements
        test_sample = self.sampler.sample(1)
        #self.optimizer= torch.optim.Adam(self.q_value_net.parameters(), lr=0.001)
        discount = self.mdp.discount
        for e in range(self.iterations):

            states = self.sampler.sample(batch_size)
            rates, next_states = self.mdp.t_model.rates(states[None, ...], actions[:, None, :])
            exit_rates = rates.sum(0)
            self.optimizer.zero_grad()

            Q_current = self.q_value_net(states.float()/self.normalization)
            Q_current_target = self.q_target_net(states.float()/self.normalization)

            Q_next = self.q_value_net(next_states.float()/self.normalization)
            Q_next = self.q_target_net(next_states.float() / self.normalization).detach()
            V_next, _ = Q_next.max(dim=-1)

            # Option 1 (standard Bellman equation, it has Q_current on both sides)
            V_current, _ = Q_current.max(dim=-1)
            # Option 2 (from fixed point iteration, does not depend on Q_current)
            #V_current,_ = ((self.mdp.r_model(states[None, ...], actions[:, None, ...]) + discount * (rates * V_next).sum(0))/(1 + discount * exit_rates)).max(dim=0)


            target = self.mdp.r_model(states[None, ...], actions[:, None, ...]) + discount * (rates * V_next).sum(0)
            target+= - discount * exit_rates * V_current

            #
            # Q_next = self.q_target_net(next_states.float()/self.normalization).detach()
            # #Maximize over action dimension
            # V_next,_ = Q_next.max(dim=-1)
            # target = (self.mdp.r_model(states[None, ...], actions[:, None, ...]) / self.mdp.discount +
            #           (rates * V_next).sum(0)) / (1 / self.mdp.discount + exit_rates)

            #Transpose target, so it has the form n_states x n_actions
            loss = (((Q_current - target.transpose(1,0)))**2).mean()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()


            target_net_state_dict = self.q_target_net.state_dict()
            value_net_state_dict = self.q_value_net.state_dict()

            for key in value_net_state_dict:
                target_net_state_dict[key] = value_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            self.q_target_net.load_state_dict(target_net_state_dict)

            if e%1000==0:
                print("Value_net solving iteration: {}. loss : {}".format(e,loss))
                print("test_batch: {}. loss:{}".format(test_sample,self.q_value_net(test_sample/self.normalization)))


        return self.q_value_net