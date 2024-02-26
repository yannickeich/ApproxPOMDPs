from packages.model.components.spaces import DiscreteSpace, FiniteDiscreteSpace
from packages.model.components.transition_model import CRNTransitionModel
from packages.model.components.reward_model import RewardModel
from packages.model.components.observation_model import ExactContTimeObservationModel
from packages.model.pomdp import DiscreteStateContTimePOMDP
from packages.model.components.filter.cont_time_obs.cont_time_filter import ContTimeProjectionFilter
from packages.model.mdp import DiscreteMDP
from packages.model.components.agents.projection_agent import ContTimeProjectionAgent
import torch



##Network with X1,X2,X3,X4
#cyclic form X1 <-> X2 <-> X4 <-> X3 <-> X1
# where X3 and X4 are observed. Total amount of species stays constant


class ContTime4DSSpace(DiscreteSpace):
    def __init__(self):
        """
        State space for a ContTime CRN

        """
        super().__init__(4)


class ContTime4DASpace(FiniteDiscreteSpace):
    def __init__(self):
        """
        Action space for the ContTime CRN (one rate can be changed with action 0 or action 1)
        """
        super().__init__(cardinalities=(2,))


class ContTime4DTransitionModel(CRNTransitionModel):
    s_space: ContTime4DSSpace
    a_space: ContTime4DASpace

    def __init__(self, s_space: ContTime4DSSpace, a_space: ContTime4DASpace,
                 S: torch.Tensor = torch.tensor([[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]]),
                 P: torch.Tensor = torch.tensor([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]]),
                 c: torch.Tensor = torch.tensor([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]).T):
        """
        Transition model for the controlled  CRN
        change vectors are the same for all actions
        action decides the rate coefficients. (action 0 takes the first column etc.)
        :param s_space:
        :param a_space:
        :param S: stoichiometric substrate coefficients
        :param P: stoichiometric product coefficients
        :param c: rate coefficients (action
        """

        super().__init__(s_space, a_space,S,P,c)



class ContTime4DOSpace(DiscreteSpace):
    def __init__(self):
        """
        Observation space of the Lotka Volterra CRN
        """

        super().__init__(2)


class ContTime4DObservationModel(ExactContTimeObservationModel):
    s_space: ContTime4DSSpace
    a_space: ContTime4DASpace
    o_space: ContTime4DOSpace

    def __call__(self, state: torch.Tensor):
        #returns the observed states X3 and X4
        return state[...,2:]



class ContTime4DRewardModel(RewardModel):
    def __call__(self, state: torch.Tensor, action: torch.Tensor,**kwargs):
        """
        reward model for the ContTime4D network --> state in equilibrium -> punish maximum state

        :param state: 3d queueing state
        :param action: action signal
        :return: negative number of all packets
        """
        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        max_state,_=state.max(dim=-1)
        reward = - max_state

        return reward.expand(batch_size)


class ContTime4DPOMDP(DiscreteStateContTimePOMDP):
    s_space: ContTime4DSSpace
    a_space: ContTime4DASpace
    o_space: ContTime4DOSpace
    t_model: ContTime4DTransitionModel
    o_model: ContTime4DObservationModel
    r_model: ContTime4DRewardModel

    def __init__(self,
                 S: torch.Tensor = torch.tensor(
                     [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                      [0, 0, 0, 1]]),
                 P: torch.Tensor = torch.tensor(
                     [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0],
                      [0, 0, 1, 0]]),
                 c: torch.Tensor = torch.tensor(
                     [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).T,discount=1/-torch.log(torch.tensor(0.9)),device=None):
        """
        3D queueing network POMDP

        :param buffer_sizes: sizes of the three buffers
        :param arrival_rates: arrival rates for queue 0 and queue 1
        :param service_rates: service rates for queue 0, queue 1 and queue 2
        :param obs_rate:  Poisson rate parameter for discrete time observation
        :param success_prob: Binomial success probability for observing a packet in queue 2
        :param discount: POMDP discount
        """
        s_space = ContTime4DSSpace()
        a_space = ContTime4DASpace()
        o_space = ContTime4DOSpace()

        t_model = ContTime4DTransitionModel(s_space=s_space, a_space=a_space, S=S.to(device), P=P.to(device), c=c.to(device))
        o_model = ContTime4DObservationModel(s_space=s_space, a_space=a_space, o_space=o_space)
        r_model = ContTime4DRewardModel()

        super().__init__(a_space, s_space, o_space, t_model, o_model, r_model, discount)


class ContTime4DMDP(DiscreteMDP):
    s_space: ContTime4DSSpace
    a_space: ContTime4DASpace
    t_model: ContTime4DTransitionModel
    r_model: ContTime4DRewardModel

    def __init__(self,  S: torch.Tensor = torch.tensor([[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]]),
                 P: torch.Tensor = torch.tensor([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]]),
                 c: torch.Tensor = torch.tensor([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]).T,
                 discount=1/-torch.log(torch.tensor(0.9)),device=None):
        """
        3D queueing network MDP

        :param buffer_sizes: sizes of the three buffers
        :param arrival_rates: arrival rates for queue 0 and queue 1
        :param service_rates: service rates for queue 0, queue 1 and queue 2
        :param discount: MDP discount
        """
        s_space = ContTime4DSSpace()
        a_space = ContTime4DASpace()

        t_model = ContTime4DTransitionModel(s_space=s_space, a_space=a_space, S=S.to(device), P=P.to(device), c=c.to(device))
        r_model = ContTime4DRewardModel()

        super().__init__(a_space, s_space, t_model, r_model, discount)


class Multinomial4DFilter(ContTimeProjectionFilter):
    t_model: CRNTransitionModel
    o_model: ExactContTimeObservationModel

    def __init__(self, t_model: CRNTransitionModel, o_model: ExactContTimeObservationModel, initial_param: torch.Tensor,
                 initial_time: torch.Tensor, exp_method='MC', exp_samples=1000, device=None, sim_method=None,
                 sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Base class for a projection type filter using a discrete parametrization
        :param t_model:
        :param o_model:
        :param initial_param:
        :param initial_time:
        :param device:
        """

        ContTimeProjectionFilter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param,
                                  initial_time=initial_time, exp_method=exp_method, exp_samples=exp_samples,
                                  device=device, sim_method=sim_method, sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)

    def drift(self, belief, action,lambda_ = 1e-6, **kwargs):
        theta1 = belief
        theta2 = 1-belief
        rates = self.t_model.c
        drift = -rates[0,action] * theta1 + rates[2,action] * theta2 - rates[1,action] * theta1 + rates[1,action] * theta1 * theta1 + rates[2,action] * theta2 * theta1
        return drift

    def jump_update(self, param, observation, action, **kwargs):
        ## all possible change vectors
        change_vectors = self.o_model(self.t_model.P - self.t_model.S)
        ##compare change vectors with observation:
        idx = (observation == change_vectors).prod(dim=-1)==1
        possible_change_vectors = change_vectors[idx]
        rates = self.t_model.c[idx,]
        return observation

    def log_prob(self, state: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def sample(self, param, num_samples=1):
        raise NotImplementedError
    def moments(self, param: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError



class Multinomial4DAgent(Multinomial4DFilter, ContTimeProjectionAgent):
    t_model: CRNTransitionModel
    o_model = ExactContTimeObservationModel

    def __init__(self, t_model: CRNTransitionModel,
                 o_model: ExactContTimeObservationModel,
                 initial_param: torch.Tensor,
                 initial_time: torch.Tensor,
                 advantage_net=None, exp_method='MC', exp_samples=1000, device=None, sim_method=None, sim_options=None,
                 jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        Multinomial4DFilter.__init__(self, t_model, o_model, initial_param, initial_time, exp_method=exp_method,
                                          exp_samples=exp_samples, device=device, sim_method=sim_method,
                                          sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)
        ContTimeProjectionAgent.__init__(self, t_model, o_model, initial_param, initial_time, advantage_net,
                                 exp_method=exp_method, exp_samples=exp_samples, device=device, sim_method=sim_method,
                                 sim_options=sim_options,
                                  jump_optim=jump_optim,jump_opt_iter=jump_opt_iter,jump_opt_options=jump_opt_options)