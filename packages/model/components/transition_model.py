from abc import ABC, abstractmethod
from packages.model.components.spaces import DiscreteSpace, Space, FiniteDiscreteSpace, ContinuousSpace
import torch
import numpy as np


class TransitionModel(ABC):
    def __init__(self, s_space: Space, a_space: Space):
        """
        Base class for a transition model

        :param s_space: state space
        :param a_space: action space
        """
        self.s_space = s_space
        self.a_space = a_space


class DiscreteTransitionModel(TransitionModel, ABC):
    s_space: DiscreteSpace

    def __init__(self, s_space: DiscreteSpace, a_space: Space):
        """
        Implements a transition model for a countable state space

        :param s_space: state space
        :param a_space: action space
        """
        super().__init__(s_space=s_space, a_space=a_space)

    def sample_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Samples next state for state and action (can be batches)

        :param state: [state_batch, state_dim] or [state_dim]
        :param action: [action_batch, action_dim] or [action_dim]
        :return: next_state: [state_batch, action_batch, state_dim]
        """
        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])

        # Get all next states with rates
        rates, next_state = self.rates(state, action)
        # rates Dimension num_events x state_batch x action_batch
        # next_state, Dimension num_events x state_batch x action_batch x state_dim

        num_events = next_state.shape[0]
        state_dim = next_state.shape[-1]

        # Draw events from rates
        weights = rates.view(num_events, -1).T  # Dimension state_batch * action_batch x num_events
        event = (torch.multinomial(weights, 1, replacement=True)[..., 0]).reshape(
            batch_size)  # Dimension state_batch x action_batch

        # Gather the next states given the events
        index = event[None, ..., None].expand(1, *batch_size,
                                              state_dim)  # Dimension 1 x state_batch x action_batch x state_dim
        next_state = torch.gather(next_state, dim=0, index=index)[
            0, ...]  # Dimension state_batch x action_batch x state_dim

        return next_state

    def exit_rate(self, state: torch.Tensor, action: torch.Tensor, keepdim=False) -> torch.Tensor:
        """
        Computes the exit rates for state and action (can be batches)

        :param state: [state_batch, state_dim] or [state_dim]
        :param action: [action_batch, action_dim] or [action_dim]
        :param keepdim: keep singleton dimensions
        :return: exit rate [state_batch, action_batch] squeezed if keepdim false)
        """
        rates, _ = self.rates(state, action)
        # dim num_events x batch_size
        exit_rate = rates.sum(dim=0)

        return exit_rate

    def max_exit_rate(self, state: torch.Tensor, keepdim=False):
        """
        Computes the maximum (over actions) exit rate for the given state
        :param state: [state_batch, state_dim] or [state_dim]
        :param keepdim: keep singleton dimensions
        :return: maximum exit rate [state_batch] or [1]
        """
        if isinstance(self.a_space, FiniteDiscreteSpace):
            action = self.a_space.elements
            # add action batch to state
            exit_rate = self.exit_rate(state[None,...], action, keepdim=keepdim)
            max_exit_rate = torch.max(exit_rate, dim=0).values
            return max_exit_rate

        else:
            # TODO implement?
            raise NotImplementedError

    @abstractmethod
    def rates(self, state: torch.Tensor, action: torch.Tensor):
        """
        Abstract method which returns rates and next states
        """
        raise NotImplementedError


class CRNTransitionModel(DiscreteTransitionModel):
    s_space: DiscreteSpace

    def __init__(self,s_space:DiscreteSpace,a_space: Space, S: torch.Tensor, P: torch.Tensor, c:torch.Tensor):
        """
        Implements a Transition Model for chemical reaction networks
        :param s_space:
        :param a_space:
        :param S: stoichiometric substrate coefficients
        :param P: stoichiometric product coefficients
        :param c: reaction rates
        """
        super().__init__(s_space=s_space,a_space=a_space)
        # Check that substrate and coefficient matrix have the same size
        assert(S.shape == P.shape)
        # Check that the state dimensions of the matrices is the same as of the state space
        assert(s_space.dimensions == P.shape[-1])
        # Check that the number of reactions is the same for the reaction rates and the
        assert(c.shape[0] == S.shape[0])
        assert(c.shape[1]==a_space.elements.shape[0])
        self.S = S
        self.P = P
        self.c = c


    def rates(self, state: torch.Tensor, action: torch.Tensor):
        """
        Computes the next states and the rates of the CTMC given the current state and action.
        Overwrite this method in each CRN class to specify how the actions effect the rates.
        This is just a sample, where the actions have no effect.
        Notice the difference between rates (rates of the CTMC, prospensities) and c (reaction rates, constants that are needed to compute the rates)
        :param state:
        :param action:
        :return: rates, next_states
        """
        ## TODO compute prospensities out of reaction rates, state and S, and action
        ## TODO be careful to use the correct formula, depending on the reaction rates c being the macroscopic rates or not

        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        state = state.expand(*batch_size,state.shape[-1])
        # compute rates with principle of mass action
        c = self.c[:,action].squeeze(-1)
        S = self.S
        P = self.P
        S = S.expand(*batch_size, S.shape[0], S.shape[1]).permute(-2, *(range(len(batch_size))), -1)
        P = P.expand(*batch_size, P.shape[0], P.shape[1]).permute(-2, *(range(len(batch_size))), -1)
        # dimensions: S [num_events, *batch_size, state_dim], c: [num_events, *batch_size], state [batch_size, state_dim]
        rates = c * (torch.lgamma(state[None]+1)-torch.lgamma(S +1)-torch.lgamma(state[None]-S +1)).exp().prod(-1) #dimensions: num_events x *batch_size

        change_vectors = P-S # dimension num_events x batch_size x state_dim
        next_states = (state[None] + change_vectors).clip(min=0.0) # Dimension num_events x batch_size x state_dim

        return rates, next_states


class TruncCRNTransitionModel(DiscreteTransitionModel):
    s_space: FiniteDiscreteSpace

    def __init__(self,s_space:FiniteDiscreteSpace,a_space: Space, S: torch.Tensor, P: torch.Tensor, c:torch.Tensor):
        """
        Implements a Transition Model for chemical reaction networks
        :param s_space:
        :param a_space:
        :param S: stoichiometric substrate coefficients
        :param P: stoichiometric product coefficients
        :param c: reaction rates
        """
        super().__init__(s_space=s_space,a_space=a_space)
        # Check that substrate and coefficient matrix have the same size
        assert(S.shape == P.shape)
        # Check that the state dimensions of the matrices is the same as of the state space
        assert(s_space.dimensions == P.shape[-1])
        # Check that the number of reactions is the same for the reaction rates and the
        assert(c.shape[0] == S.shape[0])
        assert(c.shape[1]==a_space.elements.shape[0])
        self.S = S
        self.P = P
        self.c = c


    def rates(self, state: torch.Tensor, action: torch.Tensor):
        """
        Computes the next states and the rates of the CTMC given the current state and action.
        Overwrite this method in each CRN class to specify how the actions effect the rates.
        This is just a sample, where the actions have no effect.
        Notice the difference between rates (rates of the CTMC, prospensities) and c (reaction rates, constants that are needed to compute the rates)
        :param state:
        :param action:
        :return: rates, next_states
        """
        ## TODO compute prospensities out of reaction rates, state and S, and action
        ## TODO be careful to use the correct formula, depending on the reaction rates c being the macroscopic rates or not

        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        state = state.expand(*batch_size,state.shape[-1])
        # compute rates with principle of mass action
        #TODO check that dimension of c is correct for batches
        c = self.c[:,action].squeeze(-1)
        S = self.S
        P = self.P
        S = S.expand(*batch_size, S.shape[0], S.shape[1]).permute(-2, *(range(len(batch_size))), -1)
        P = P.expand(*batch_size, P.shape[0], P.shape[1]).permute(-2, *(range(len(batch_size))), -1)
        # dimensions: S [num_events, *batch_size, state_dim], c: [num_events, *batch_size], state [batch_size, state_dim]
        rates = c * (torch.lgamma(state[None]+1)-torch.lgamma(S +1)-torch.lgamma(state[None]-S +1)).exp().prod(-1) #dimensions: num_events x *batch_size

        change_vectors = P-S # dimension num_events x batch_size x state_dim
        next_states = (state[None] + change_vectors).clip(min=0.0).clip(max=torch.tensor(self.s_space.cardinalities)-1) # Dimension num_events x batch_size x state_dim

        return rates, next_states

class MultinomialMonoCRNTransitionModel(DiscreteTransitionModel):
    s_space: DiscreteSpace

    def __init__(self,s_space:DiscreteSpace,a_space: Space, c:torch.Tensor, total_N):
        """
        Implements a Transition Model for chemical reaction networks
        :param s_space:
        :param a_space:
        :param c: reaction rates element ijk is the rate going from state i to state j, given action k
        """
        super().__init__(s_space=s_space,a_space=a_space)

        assert(c.shape[1]==s_space.dimensions)
        assert(c.shape[1]==c.shape[2])
        assert(c.shape[0]==a_space.elements.shape[0])
        d = c.shape[1]

        # Create change vectors. element ij is the vector that belongs to reaction ij, has -1 at position i and +1 at j
        change_vectors = torch.zeros((d,d,d),dtype=int)
        for i in range(d):
            for j in range(d):
                change_vectors[i,j,i] += -1
                change_vectors[i,j,j] += 1

        self.change_vectors = change_vectors
        self.c = c
        self.total_N = total_N


    def rates(self, state: torch.Tensor, action: torch.Tensor):
        """
        Computes the next states and the rates of the CTMC given the current state and action.
        Overwrite this method in each CRN class to specify how the actions effect the rates.
        This is just a sample, where the actions have no effect.
        Notice the difference between rates (rates of the CTMC, prospensities) and c (reaction rates, constants that are needed to compute the rates)
        :param state:
        :param action:
        :return: rates, next_states
        """

        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        state = state.expand(*batch_size,state.shape[-1])
        action = action.expand(*batch_size, action.shape[-1])
        #dim of state: *batch_size x state_dim
        #dim of action: *batch_size x action_dim(1)
        # get rates, belonging to actions
        c = self.c[action.squeeze(-1),...]
        #dim of c: *batch_size x state_dim(reaction_from) x state_dim(reaction_to)

        dim = c.shape[-1]

        rates = c * state[...,None]
        # rates dim: *batch_size x state_dim (reaction_from) x state_dim (reaction_to)
        #permute to  state_dim (reaction_from) x state_dim (reaction_to) x state_batch
        rates = rates.permute(-2,-1,*range(0, len(rates.shape[:-2])))
        rates = rates.reshape(dim*dim,*batch_size)


        next_states = (state[...,None,None,:] + self.change_vectors).clip(min=0.0) #state_batch x state_dim (reaction_from) x state_dim (reaction_to) x state_dim

        #Put states that are outside of the state space (and therefore have rate of 0) to a state in the state space
        #Will not have an effect, apart that they are in state space
        sub_state = torch.zeros(state.shape[-1],dtype=int)
        sub_state[0] = self.total_N
        next_states[(next_states.sum(-1) != self.total_N)] = sub_state

        next_states = next_states.permute(-3,-2,*range(0,len(next_states.shape[:-3])),-1)
        next_states = next_states.reshape(dim * dim, *batch_size,dim)

        #Reforming is done so that next_states has dim: num_events x *batch_size x state_dim
        # and rates has dim: num_events x *batch_size
        return rates, next_states

class PoissonMonoCRNTransitionModel(DiscreteTransitionModel):
    s_space: DiscreteSpace

    def __init__(self,s_space:DiscreteSpace,a_space: Space, c:torch.Tensor,b:torch.Tensor, d:torch.Tensor):
        """
        Implements a Transition Model for chemical reaction networks. Includes birth and death processes, compared to the multinomial case
        :param s_space:
        :param a_space:
        :param c: reaction rates element ijk is the rate going from state j to state k, given action i
        :param b: birth rates, element ij , birth rate of species j, given action i
        :param d: death rates, element ij is the death rate of species j, given action i
        """
        super().__init__(s_space=s_space,a_space=a_space)
        assert(b.shape[1]==s_space.dimensions)
        assert(d.shape[1]==s_space.dimensions)
        assert(c.shape[1]==s_space.dimensions)
        assert(c.shape[1]==c.shape[2])
        assert(c.shape[0]==a_space.elements.shape[0])
        dim = c.shape[1]

        # Create change vectors.
        #Conversion change vectors
        # element ij is the vector that belongs to reaction ij, has -1 at position i and +1 at j ( not action dependent)
        c_change_vectors = torch.zeros((dim,dim,dim),dtype=int)
        for i in range(dim):
            for j in range(dim):
                c_change_vectors[i,j,i] += -1
                c_change_vectors[i,j,j] += 1
        #Birth change vectors:
        b_change_vectors = torch.diag(torch.ones(dim,dtype=int))
        #Death change_vectors:
        d_change_vectors = -torch.diag(torch.ones(dim, dtype=int))
        self.c_change_vectors = c_change_vectors
        self.b_change_vectors = b_change_vectors
        self.d_change_vectors = d_change_vectors
        self.c = c
        self.b = b
        self.d = d


    def rates(self, state: torch.Tensor, action: torch.Tensor):
        """
        Computes the next states and the rates of the CTMC given the current state and action.
        Overwrite this method in each CRN class to specify how the actions effect the rates.
        This is just a sample, where the actions have no effect.
        Notice the difference between rates (rates of the CTMC, prospensities) and c (reaction rates, constants that are needed to compute the rates)
        :param state:
        :param action:
        :return: rates, next_states
        """

        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        state = state.expand(*batch_size,state.shape[-1])
        action = action.expand(*batch_size, action.shape[-1])
        #dim of state: *batch_size x state_dim
        #dim of action: *batch_size x action_dim(1)
        # get rates, belonging to actions
        c = self.c[action.squeeze(-1),...]
        b = self.b[action.squeeze(-1),...]
        d = self.d[action.squeeze(-1),...]

        ### Rates and Next states of conversion reactions
        #dim of c: *batch_size x state_dim(reaction_from) x state_dim(reaction_to)
        dim = c.shape[-1]

        c_rates = c * state[...,None]
        # rates dim: *batch_size x state_dim (reaction_from) x state_dim (reaction_to)
        #permute to  state_dim (reaction_from) x state_dim (reaction_to) x batch_size
        c_rates = c_rates.permute(-2,-1,*range(0, len(c_rates.shape[:-2])))
        c_rates = c_rates.reshape(dim*dim,*batch_size)


        c_next_states = (state[...,None,None,:] + self.c_change_vectors).clip(min=0.0) #state_batch x state_dim (reaction_from) x state_dim (reaction_to) x state_dim

        c_next_states = c_next_states.permute(-3,-2,*range(0,len(c_next_states.shape[:-3])),-1)
        c_next_states = c_next_states.reshape(dim * dim, *batch_size,dim)

        #Reforming is done so that next_states has dim: num_events x *batch_size x state_dim
        # and rates has dim: num_events x *batch_size

        ### Rates and next_states of death reactions
        # dim of d: *batch_size x state_dim(reaction_from)

        d_rates = d * state
        d_rates = d_rates.permute(-1,*range(0,len(d_rates.shape[:-2])))

        d_next_states = (state[...,None,:] + self.d_change_vectors).clip(min=0.0)  #state_batch x state_dim (reaction_from)  x state_dim
        d_next_states = d_next_states.permute(-2,*range(0,len(d_next_states.shape[:-2])),-1)

        ### Rates and next_states of Birth reactions
        # dim of b: *batch_size x state_dim(reaction_to)

        b_rates = b
        b_rates = b_rates.permute(-1, *range(0, len(b_rates.shape[:-2])))

        b_next_states = (state[..., None, :] + self.b_change_vectors).clip(
            min=0.0)  # state_batch x state_dim (reaction_from)  x state_dim
        b_next_states = b_next_states.permute(-2, *range(0, len(b_next_states.shape[:-2])), -1)

        rates = torch.cat((c_rates,d_rates,b_rates))
        next_states = torch.cat((c_next_states,d_next_states,b_next_states))

        return rates, next_states

class QueueingTransitionModel(DiscreteTransitionModel):
    s_space: FiniteDiscreteSpace

    def __init__(self, s_space: FiniteDiscreteSpace, a_space: Space, birth: torch.Tensor, death: torch.Tensor,
                 inter_rates: torch.Tensor):
        """
        Implements a Transition model for queueing systems.
        :param s_space:
        :param a_space:
        :param birth: arrival rates from no other queue    action_dim x state_dim
        :param death: service rates to no other queue      action_dim x state_dim
        :param inter_rates: connected rates that are being serviced by queue i and arrive at queue j  action_dim x state_dim x state_dim
        """
        super().__init__(s_space=s_space, a_space=a_space)
        assert (s_space.dimensions == birth.shape[-1])
        assert (death.shape == birth.shape)
        assert (s_space.dimensions == inter_rates.shape[-1])
        assert (a_space.elements.shape[0] == inter_rates.shape[0])
        self.birth_rates = birth
        self.death_rates = death
        self.inter_rates = inter_rates

    def rates(self, state: torch.Tensor, action: torch.Tensor):
        # if action.ndim==1:
        #     action = action[None]

        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        state = state.expand(*batch_size, state.shape[-1])  # dim: *batch_size x state_dim
        birth_rates = self.birth_rates[action].squeeze(
            -2)  # dim: *batch_size x state_dim.  Squeezes the action dim (action is batch_size x action_dim(1))
        birth_rates = birth_rates * (state < self.s_space.buffer_sizes)
        birth_next_states = state[..., None, :] + torch.diag(
            torch.ones(state.shape[-1], dtype=int))  # dim: *batch_size x num_events(state_dim) x state_dim

        death_rates = self.death_rates[action].squeeze(
            -2)  # dim: *batch_size x state_dim.  Squeezes the action dim (action is batch_size x action_dim(1))
        death_rates = death_rates * (state > 0)
        death_next_states = state[..., None, :] - torch.diag(
            torch.ones(state.shape[-1], dtype=int))  # dim: *batch_size x num_events(state_dim) x state_dim

        inter_rates = self.inter_rates[action].squeeze(
            -3)  # dim: *batch_size x state_dim x state_dim.  Squeezes the action dim (action is batch_size x action_dim(1))
        # Check if token can arrive and if there is one to leave
        inter_rates = (inter_rates * (state < (self.s_space.buffer_sizes))[..., None, :] * (state > 0)[
            ..., None]).reshape(*batch_size, -1)  # dim: batch_size x num_events(state_dim x state_dim)
        inter_next_states = (
                    state[..., None, None, :] + torch.diag((torch.ones(state.shape[-1], dtype=int))) - torch.diag(
                (torch.ones(state.shape[-1], dtype=int)))[..., None, :]).reshape(*batch_size, -1, state.shape[
            -1])  # dim: batch_size x num_events x state_dim

        rates = torch.cat((birth_rates, death_rates, inter_rates), dim=-1)  # dim: *batch_size x num_events
        next_states = torch.cat((birth_next_states, death_next_states, inter_next_states), dim=-2).clip(
            min=0.0).clip(max=torch.tensor(self.s_space.cardinalities) - 1)  # dim: *batch_size x num_events x dim

        # Permute num_events dimension to the front
        return rates.permute(-1, *range(len(batch_size))), next_states.permute(-2, *range(len(batch_size)), -1)


class TruncPoissonMonoCRNTransitionModel(DiscreteTransitionModel):
    s_space: FiniteDiscreteSpace

    def __init__(self, s_space: FiniteDiscreteSpace, a_space: Space, c: torch.Tensor, b: torch.Tensor, d: torch.Tensor):
        """
        Implements a Transition Model for chemical reaction networks. Includes birth and death processes, compared to the multinomial case
        :param s_space:
        :param a_space:
        :param c: reaction rates element ijk is the rate going from state j to state k, given action i
        :param b: birth rates, element ij , birth rate of species j, given action i
        :param d: death rates, element ij is the death rate of species j, given action i
        """
        super().__init__(s_space=s_space, a_space=a_space)
        assert (b.shape[1] == s_space.dimensions)
        assert (d.shape[1] == s_space.dimensions)
        assert (c.shape[1] == s_space.dimensions)
        assert (c.shape[1] == c.shape[2])
        assert (c.shape[0] == a_space.elements.shape[0])
        dim = c.shape[1]

        # Create change vectors.
        # Conversion change vectors
        # element ij is the vector that belongs to reaction ij, has -1 at position i and +1 at j ( not action dependent)
        c_change_vectors = torch.zeros((dim, dim, dim), dtype=int)
        for i in range(dim):
            for j in range(dim):
                c_change_vectors[i, j, i] += -1
                c_change_vectors[i, j, j] += 1
        # Birth change vectors:
        b_change_vectors = torch.diag(torch.ones(dim, dtype=int))
        # Death change_vectors:
        d_change_vectors = -torch.diag(torch.ones(dim, dtype=int))
        self.c_change_vectors = c_change_vectors
        self.b_change_vectors = b_change_vectors
        self.d_change_vectors = d_change_vectors
        self.c = c
        self.b = b
        self.d = d

    def rates(self, state: torch.Tensor, action: torch.Tensor):
        """
        Computes the next states and the rates of the CTMC given the current state and action.
        Overwrite this method in each CRN class to specify how the actions effect the rates.
        This is just a sample, where the actions have no effect.
        Notice the difference between rates (rates of the CTMC, prospensities) and c (reaction rates, constants that are needed to compute the rates)
        :param state:
        :param action:
        :return: rates, next_states
        """

        batch_size = torch.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        state = state.expand(*batch_size, state.shape[-1])
        action = action.expand(*batch_size, action.shape[-1])
        # dim of state: *batch_size x state_dim
        # dim of action: *batch_size x action_dim(1)
        # get rates, belonging to actions
        c = self.c[action.squeeze(-1), ...]
        b = self.b[action.squeeze(-1), ...]
        d = self.d[action.squeeze(-1), ...]

        ### Rates and Next states of conversion reactions
        # dim of c: *batch_size x state_dim(reaction_from) x state_dim(reaction_to)
        dim = c.shape[-1]

        c_rates = c * state[..., None]
        # rates dim: *batch_size x state_dim (reaction_from) x state_dim (reaction_to)
        # permute to  state_dim (reaction_from) x state_dim (reaction_to) x batch_size
        c_rates = c_rates.permute(-2, -1, *range(0, len(c_rates.shape[:-2])))
        c_rates = c_rates.reshape(dim * dim, *batch_size)

        c_next_states = (state[..., None, None, :] + self.c_change_vectors).clip(
            min=0.0).clip(max=torch.tensor(self.s_space.cardinalities)-1)  # state_batch x state_dim (reaction_from) x state_dim (reaction_to) x state_dim

        c_next_states = c_next_states.permute(-3, -2, *range(0, len(c_next_states.shape[:-3])), -1)
        c_next_states = c_next_states.reshape(dim * dim, *batch_size, dim)

        # Reforming is done so that next_states has dim: num_events x *batch_size x state_dim
        # and rates has dim: num_events x *batch_size

        ### Rates and next_states of death reactions
        # dim of d: *batch_size x state_dim(reaction_from)

        d_rates = d * state
        d_rates = d_rates.permute(-1, *range(0, len(d_rates.shape[:-1])))

        d_next_states = (state[..., None, :] + self.d_change_vectors).clip(
            min=0.0).clip(max=torch.tensor(self.s_space.cardinalities)-1)  # state_batch x state_dim (reaction_from)  x state_dim
        d_next_states = d_next_states.permute(-2, *range(0, len(d_next_states.shape[:-2])), -1)

        ### Rates and next_states of Birth reactions
        # dim of b: *batch_size x state_dim(reaction_to)

        b_rates = b
        b_rates = b_rates.permute(-1, *range(0, len(b_rates.shape[:-1])))

        b_next_states = (state[..., None, :] + self.b_change_vectors).clip(
            min=0.0).clip(max=torch.tensor(self.s_space.cardinalities)-1)  # state_batch x state_dim (reaction_from)  x state_dim
        b_next_states = b_next_states.permute(-2, *range(0, len(b_next_states.shape[:-2])), -1)

        rates = torch.cat((c_rates, d_rates, b_rates))
        next_states = torch.cat((c_next_states, d_next_states, b_next_states))

        return rates, next_states
class TabularTransitionModel(DiscreteTransitionModel):
    s_space: FiniteDiscreteSpace
    a_space: FiniteDiscreteSpace

    def __init__(self, s_space: FiniteDiscreteSpace, a_space: FiniteDiscreteSpace,device=None):
        """
        Class for finite state action space transition model which is represented by a table
        :param s_space: finite state space
        :param a_space: finite action space
        """
        super().__init__(s_space=s_space, a_space=a_space)
        states = self.s_space.elements.to(device)
        actions = self.a_space.elements.to(device)
        self.n_states = states.shape[0]
        self.n_actions = actions.shape[0]

        # Compute table entries
        rates, next_states = self.rates(states[None, ...], actions[:, None, :])
        #rates dim: num_events x n_actions x n_states
        table = torch.zeros(self.n_states, self.n_states, self.n_actions).to(device)
        for action_idx in range(self.n_actions):
            # Convert next states to linear index
            lin_idx = torch.as_tensor(
                np.ravel_multi_index(next_states[:, action_idx, ...].to('cpu').flatten(end_dim=-2).numpy().T,
                                     self.s_space.cardinalities)).view_as(rates[:, action_idx, ...]).to(device)
            for state_idx in range(self.n_states):
                for event_idx in range(lin_idx.shape[0]):
                    next_state_idx = lin_idx[event_idx, state_idx]
                    # Fill transition rate table
                    table[state_idx, next_state_idx, action_idx] = rates[event_idx, action_idx, state_idx]
                table[state_idx, state_idx, action_idx] = -table[state_idx, :, action_idx].sum()
        self.table = table


# For later
class ContinuousTransitionModel(TransitionModel, ABC):
    def __init__(self, s_space: ContinuousSpace, a_space: Space):
        """
        Implements a transition model for continuous state space
        :param s_space:
        :param a_space:
        """
        super().__init__(s_space=s_space, a_space=a_space)

    def drift(self, state: torch.Tensor, action: torch.Tensor):
        """
        Abstract method which returns the drift of the sde
        :param state:
        :param action:
        :return:
        """
        raise NotImplementedError

    def dispersion(self, state: torch.Tensor, action: torch.Tensor):
        """
        Abstract method which returns dispersion of the sde
        :param state:
        :param action:
        :return:
        """
        raise NotImplementedError
