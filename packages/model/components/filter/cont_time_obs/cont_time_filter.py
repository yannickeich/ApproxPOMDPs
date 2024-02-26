from packages.model.components.filter.filter import Filter
from packages.model.components.transition_model import TransitionModel, MultinomialMonoCRNTransitionModel, PoissonMonoCRNTransitionModel, TruncPoissonMonoCRNTransitionModel
from packages.model.components.observation_model import ExactContTimeObservationModel
import torch
from abc import ABC, abstractmethod

EPS = 1e-5



class ContTimeProjectionFilter(Filter, ABC):
    o_model: ExactContTimeObservationModel

    def __init__(self, t_model: TransitionModel, o_model: ExactContTimeObservationModel, initial_param: torch.Tensor,
                 initial_time: torch.Tensor, exp_method='MC', exp_samples=1000, device=None, sim_method=None,
                 sim_options=None,jump_optim=torch.optim.Adam,jump_opt_iter=500,jump_opt_options = {"lr":0.01}):
        """
        Base Class for a projection type filter using a finite parametrization

        :param t_model: transition model
        :param o_model: observation model
        :param initial_param: initial parameter for simulation
        :param initial_time: initial time for simulation
        :param exp_method: method used for expectation of functions, if not specified otherwise
        """
        self.device = device
        self.exp_method = exp_method
        self.exp_samples = exp_samples
        self.jump_optim = jump_optim
        self.jump_opt_iter = jump_opt_iter
        self.jump_opt_options = jump_opt_options
        Filter.__init__(self, t_model=t_model, o_model=o_model, initial_param=initial_param, initial_time=initial_time,
                        sim_method=sim_method, sim_options=sim_options)

    @abstractmethod
    def moments(self, param: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns mean, variance and skew of the filter distribution
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, state: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability for the Filtering distribution

        :param state: state to be evaluated
        :param param: parameter of the filtering distribution
        :return: log probability
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, param, num_samples=1):
        """
        Creates Samples from the filter distribution

        :param param: parameter of the filter
        :param num_samples: number of samples
        """
        raise NotImplementedError


    def grad_log_prob(self, state: torch.Tensor, param: torch.Tensor, **kwargs):
        """
        Computes the gradient of the log probability for the Filtering distribution w.r.t. the belief parameter.
        Keyword arguments can contain create_graph and retain_graph for autograd function.

        :param state:  state to be evaluated
        :param param: parameter of the filtering distribution
        :return: gradient of the log probability
        """

        batch_size = torch.broadcast_shapes(state.shape[:-1], param.shape[:-1])
        param = param.expand(*batch_size, param.shape[-1])
        param.requires_grad_(True)
        log_prob = self.log_prob(state, param)
        grad = torch.autograd.grad(outputs=log_prob.sum(),
                                   inputs=param,
                                   **kwargs)[0]

        return grad

    def hessian_log_prob(self, state: torch.Tensor, param: torch.Tensor, **kwargs):
        """
        Computes the hessian of the log probability for the filtering distribution w.r.t. the belief parameter.
        Usually needed for the Fisher matrix.

        :param state:  state to be evaluated
        :param param: parameter of the filtering distribution
        :return: hessian of the log probability
        """

        batch_size = torch.broadcast_shapes(state.shape[:-1], param.shape[:-1])
        param = param.expand(*batch_size, param.shape[-1])
        param.requires_grad_(True)


        hessian = torch.autograd.functional.jacobian(
            lambda x: self.grad_log_prob(state, x, create_graph=True, retain_graph=True).reshape(-1,
                                                                                                 param.shape[-1]).sum(
                0), inputs=param,vectorize=True).permute(
            *range(1, len(state.shape[:-1]) + 1), 0, -1)


        return hessian

    def fisher_matrix(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
        """
         Computes the Fisher Matrix of the filtering distribution distribution

        :param param: parameter of the filter distribution
        :return: fisher matrix
        """

        # If not called otherwise, use filters exp method
        try:
            method = kwargs['method']
        except KeyError:
            method = self.exp_method
        try:
            num_samples = kwargs['num_samples']
        except KeyError:
            num_samples = self.exp_samples

        def func(state):
            grad_log_prob = self.grad_log_prob(state, param[..., None, :])
            return (grad_log_prob[..., None, :] * grad_log_prob[..., :, None]).permute(-2, -1, *range(
                len(grad_log_prob.shape[:-1])))

        def func2(state):
            hessian = self.hessian_log_prob(state,param[...,None,:])
            ## Bring state dimensions to the front, so that batch_size dims are in the back. State_batch needs to be last for expectation function.
            return hessian.permute(-2,-1,*range(len(hessian.shape[:-2])))
        #Different methods: product of gradients or hessian
        # gradients are faster, expectation of hessian has less variance
        #fisher_matrix = self.expectation_state_func(func, param, method=method, num_samples=num_samples)
        fisher_matrix = - self.expectation_state_func(func2, param, method=method, num_samples=num_samples)

        ## Bring state dimensions again to the last two positions
        return fisher_matrix.permute(*range(2, len(fisher_matrix.shape[:-2]) + 2), 0, 1)

    def inv_fisher_matrix(self, param: torch.Tensor, lambda_=1e-6, **kwargs) -> torch.Tensor:
        """
         Computes the inverse Fisher Matrix of the filtering distribution distribution

        :param param: parameter of the filter distribution
        :param num_samples: number of samples used for sample average
        :param lambda_: regularization parameter (numerical stability for inverse)
        :return: inverse fisher matrix
        """

        return torch.inverse(self.fisher_matrix(param, **kwargs) + lambda_ * torch.diag_embed(torch.ones_like(param)))

    def jump_update(self, param, observation, action, **kwargs):
        """
        Updates the belief given a jump in the observed states.

        :param param: initial param
        :param observation: observation used for the update
        :param action: action used at that time
        :return: new parameter after update
        """

        raise NotImplementedError


    def expectation_state_func(self, function, param: torch.Tensor, method='MC', **kwargs):
        """
        Computes the expectation of the input function, under the belief given input param

        different methods: exact, unscented, unscented_round, unscented_heuristic

        exact, unscented_round etc are not working for the continous case, can be found in discrete version


        :param function:  function, of which expectation is calculated
                        ouput of function needs to be of the form ...x...x belief_batch x state_batch, because weight will be of form belief_batch x state_batch
        :param param: parameter of pmf
        :return: expectation
        """
        if method == 'MC':
            try:
                num_samples = kwargs['num_samples']
            except KeyError:
                num_samples = 100
            # rng_state = torch.random.get_rng_state()
            # torch.random.manual_seed(hash(param))
            state = self.sample(param, num_samples)
            # torch.random.set_rng_state(rng_state)

            return function(state).mean(-1)

        elif method == 'unscented':
            # TODO make hyper-parameter accesible

            # Uses unscented methods where sigma points dont need to be in allowed state space
            # add first dimension for param batch size
            mean, var, skew = self.moments(param)

            u = torch.ones_like(mean)  # hyper-param scaling
            v = u + var ** (-1.5) * skew

            states = torch.cat([mean[..., None, :], mean[..., None, :] - torch.diag_embed(u),
                                mean[..., None, :] + torch.diag_embed(v)], dim=-2)

            w2 = (1.0 / v) / (u + v)
            w1 = (w2 * v) / u
            # w0 = 1 - torch.sum(w1) - torch.sum(w2)
            # weights = torch.cat((torch.tensor([w0]), w1, w2))
            w0 = 1.0 - w1.sum(dim=-1) - w2.sum(dim=-1)

            weights = torch.cat((w0[..., None], w1, w2), -1)
            # states = mean + torch.cat((torch.zeros_like(mean)[None,...],-torch.diag(u),torch.diag(v)))

            # weight dim belief_batch x state_batch

            return (function(states) * weights).sum(-1)

        else:
            raise NotImplementedError

    def drift(self, belief, action,lambda_ = 1e-6, **kwargs):
        """
        Compute the right hand side of the ODE for the parameter.
        :param belief:
        :param action:
        :param lambda_: regularization parameter for numerical stability when computing the inverse
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class MultinomialMonoCRNFilter(ContTimeProjectionFilter):
    o_model: ExactContTimeObservationModel
    t_model: MultinomialMonoCRNTransitionModel

    def __init__(self, t_model: MultinomialMonoCRNTransitionModel, o_model: ExactContTimeObservationModel, initial_param: torch.Tensor,
                 initial_time: torch.Tensor, exp_method='MC', exp_samples=1000, device=None, sim_method=None,
                 sim_options=None, jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}):
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
        #TODO batchwise

        #Split belief in estimate and exact_states
        exact_states = belief[...,-self.o_space.dimensions:].clone()
        theta = belief[...,:-self.o_space.dimensions].clone()
        #Add the final theta, that can be computed out of the others
        theta = torch.cat((theta,1-theta.sum(-1)[...,None]),-1)
        unobs_dim = theta.shape[-1]
        c_xx = self.t_model.c[action,:unobs_dim,:unobs_dim]
        #dtheta_i/dt = sum(j) -c_ij theta_i + c_ji theta_j
        theta_drift = -c_xx.sum(-1)*theta + (c_xx * theta[...,None]).sum(-2)
        #batch x state_dim

        c_xy = self.t_model.c[action,:unobs_dim,unobs_dim:]
        r_x = c_xy.sum(-1)

        theta_drift +=(r_x[...,None,:]*theta[...,None,:]*theta[...,None]).sum(-1) - torch.diag_embed(theta*r_x).sum(-1)
        #get rid of the last one and concatenate with zeros for the exact states
        drift = torch.cat((theta_drift[...,:-1],torch.zeros(*theta_drift.shape[:-1],self.o_space.dimensions)),-1)
        return drift

    def jump_update(self, param, observation, action, **kwargs):
        #TODO batchwise
        batch_size = torch.broadcast_shapes(param.shape[:-1], observation.shape[:-1], action.shape[:-1])
        param = param.expand(*batch_size, param.shape[-1])
        action = action.expand(*batch_size,action.shape[-1])
        observation = observation.expand(*batch_size,observation.shape[-1])

        exact_states = param[...,-self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()
        new_param = param.clone()
        check = (observation - exact_states).sum(-1)
        #check dim: *batch_size
        # if check is -1, the unobserved states increased, while the observed decreased
        # if check is 1, the observed states increased, while the unobserved decreased
        # both lead to different jump updates
        # if check is 0, the reaction was from observed to observed or from unobserved to unobserved -> no update

        #Different handling for batched and non-batched data
        ## TODO: change the jump updates for batched data. From param contains N to param containst the exact states.
        #Batched data:
        if observation.dim() > 1:
            ##Case 1
            idx_decreasing = (check == -1).nonzero(as_tuple = True)
            #index of the reactions that led to observed state decreasing
            # observation[idx_decreasing], action[idx_decreasing], param[idx_decreasing] describe these reactions and have dim: (number_of_reactions x (obs/action/param)_dim

            obs_idx = observation[idx_decreasing].nonzero(as_tuple=True)[-1]
            #obs_idx describes which of the observed states decreased
            c = self.t_model.c[action[idx_decreasing].squeeze(), param.shape[-1] + obs_idx, :param.shape[-1]]
            # rates of the observed states to the unobserved, given the action. Dim: number_of_reactions x param_dim (unobserved states)

            old_N = param[idx_decreasing][...,-1]
            # Old buffer size parameter of multinomial dist for the indexed reactions. dim: number_of_reactions
            # adding a dimension for the param_dim with None to some elements
            new_param[idx_decreasing] = (old_N[...,None] * param[idx_decreasing] + c / c.sum(-1)[...,None]) / (old_N[...,None] + 1)
            # Last element is wrong, because param[-1] contains the N parameter. Is exchanged in next step.
            # Buffer size is increased by one for this type of reactions
            new_param[idx_decreasing+(torch.tensor([-1]),)] = old_N + 1

            ### Case 2
            idx_increasing = (check == +1).nonzero(as_tuple=True)
            # index of the reactions that led to observed state increasing
            #if idx_increasing[0].size()[0]>0:
            #Next line is not working? Doing some cloning in between?
            ##new_param[idx_increasing][..., -1] -= 1
            new_param[idx_increasing+(torch.tensor([-1]),)] -=1

            ### Case 3
            idx_constant = (check == 0).nonzero(as_tuple=True)
            # index of the reactions that led to observed state staying constant. No jump update

        #Non-batched data:
        else:
            if check == 0:
                ## The observed reaction does not influence the estimated states, thetas stay the same. replace exact by obs
                new_param[...,-self.o_space.dimensions:] = observation
            elif check == 1:
                # one observed state increased. thetas stay the same, replace exact states by observation
                #new_param[...,-1] -= 1
                new_param[...,-self.o_space.dimensions:] = observation


            elif check == -1:
                # one observed state decreased. moment matching to see how it affects the estimated state. more likely rate will increase the moment more
                old_N = self.t_model.total_N - exact_states.sum(-1)
                idx_tuple = (observation - exact_states).nonzero(as_tuple=True)
                idx = idx_tuple[-1]
                c = self.t_model.c[action,param.shape[-1]+1-self.o_space.dimensions+idx,:param.shape[-1]+1-self.o_space.dimensions].squeeze()
                # Calculate new thetas (apart from the last one)
                new_theta = (old_N * theta +c[:-1]/c.sum())/(old_N+1)
                new_param = torch.cat((new_theta,observation))

            else:
                raise NotImplementedError


        return new_param

    def log_prob(self, state: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        batch_size = torch.broadcast_shapes(state.shape[:-1], param.shape[:-1])
        if len(batch_size)>=2:
            state = state.expand(*batch_size, state.shape[-1])

        exact_states_param = param[..., -self.o_space.dimensions:].clone()
        exact_states_state = state[...,-self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()
        theta = torch.cat((theta,1-theta.sum(-1)[...,None]),-1)
        N = self.t_model.total_N - exact_states_param.sum(-1)
        log_prob = torch.zeros(state.shape[:-1]).log()
        estimated_states = state[...,:-self.o_space.dimensions]
        if len(batch_size)>=2:
            check = (exact_states_state[0]==exact_states_param[0]).prod(-1)==1
            log_prob[:,check] = torch.distributions.Multinomial(int(N[0,0]),theta).log_prob(estimated_states[:,check])
        else:


        # prob is zero for all states, that do not have the same observed states as the parameter
        # Compute the log prob for the ones that have the same observed states:
            check = (exact_states_state==exact_states_param).prod(-1)==1
            log_prob[check] = torch.distributions.Multinomial(int(N),theta).log_prob(estimated_states[check])
        return log_prob

    def sample(self, param, num_samples=1):
        exact_states = param[..., -self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()
        theta = torch.cat((theta,1-theta.sum(-1)[...,None]),-1)
        N = self.t_model.total_N - exact_states.sum(-1)
        states = torch.distributions.Multinomial(int(N),theta).sample((num_samples,))
        return torch.cat((states,exact_states * torch.ones(*states.shape[:-1],exact_states.shape[-1])),-1)
    def moments(self, param: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        ## variance only contains the diagonal elements here.
        exact_states = param[...,-self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()
        N = self.t_model.total_N - exact_states.sum(-1)
        theta = torch.cat((theta,1-theta.sum(-1)[...,None]),-1)
        mean = N[...,None] * theta
        var = N[...,None] * theta * (1 - theta)
        _ = None
        return mean, var, _

class PoissonMonoCRNFilter(ContTimeProjectionFilter):
    o_model: ExactContTimeObservationModel
    t_model: PoissonMonoCRNTransitionModel

    def __init__(self, t_model: PoissonMonoCRNTransitionModel, o_model: ExactContTimeObservationModel, initial_param: torch.Tensor,
                 initial_time: torch.Tensor, exp_method='MC', exp_samples=1000, device=None, sim_method=None,
                 sim_options=None, jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}):
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
        #TODO batchwise

        #Split belief in estimate and exact_states
        exact_states = belief[...,-self.o_space.dimensions:].clone()
        theta = belief[...,:-self.o_space.dimensions].clone()

        unobs_dim = theta.shape[-1]
        c_xx = self.t_model.c[action,:unobs_dim,:unobs_dim]
        #dtheta_i/dt = sum(j) -c_ij theta_i + c_ji theta_j
        theta_drift = -c_xx.sum(-1)*theta + (c_xx * theta[...,None]).sum(-2)
        #batch x state_dim

        c_xy = self.t_model.c[action,:unobs_dim,unobs_dim:]
        r_x = c_xy.sum(-1)

        theta_drift += - (r_x+self.t_model.d[action,:unobs_dim]) * theta + self.t_model.b[action,:unobs_dim]
        # Concatenate with zeros for the exact states
        drift = torch.cat((theta_drift,torch.zeros(*theta_drift.shape[:-1],self.o_space.dimensions)),-1)
        return drift

    def jump_update(self, param, observation, action, **kwargs):
        #TODO batchwise
        batch_size = torch.broadcast_shapes(param.shape[:-1], observation.shape[:-1], action.shape[:-1])
        param = param.expand(*batch_size, param.shape[-1])
        action = action.expand(*batch_size,action.shape[-1])
        observation = observation.expand(*batch_size,observation.shape[-1])

        exact_states = param[...,-self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()
        new_param = param.clone()
        check = (observation - exact_states).sum(-1)
        #check dim: *batch_size
        # if check is -1, the unobserved states increased, while the observed decreased
        # if check is 1, the observed states increased, while the unobserved decreased
        # both lead to different jump updates
        # if check is 0, the reaction was from observed to observed or from unobserved to unobserved -> no update

        #Different handling for batched and non-batched data
        ## TODO: change the jump updates for batched data. From param contains N to param containst the exact states.
        #Batched data:
        if observation.dim() > 1:
            ##Case 1
            idx_decreasing = (check == -1).nonzero(as_tuple = True)
            #index of the reactions that led to observed state decreasing
            # observation[idx_decreasing], action[idx_decreasing], param[idx_decreasing] describe these reactions and have dim: (number_of_reactions x (obs/action/param)_dim

            obs_idx = observation[idx_decreasing].nonzero(as_tuple=True)[-1]
            #obs_idx describes which of the observed states decreased
            c = self.t_model.c[action[idx_decreasing].squeeze(), param.shape[-1] + obs_idx, :param.shape[-1]]
            # rates of the observed states to the unobserved, given the action. Dim: number_of_reactions x param_dim (unobserved states)

            old_N = param[idx_decreasing][...,-1]
            # Old buffer size parameter of multinomial dist for the indexed reactions. dim: number_of_reactions
            # adding a dimension for the param_dim with None to some elements
            new_param[idx_decreasing] = (old_N[...,None] * param[idx_decreasing] + c / c.sum(-1)[...,None]) / (old_N[...,None] + 1)
            # Last element is wrong, because param[-1] contains the N parameter. Is exchanged in next step.
            # Buffer size is increased by one for this type of reactions
            new_param[idx_decreasing+(torch.tensor([-1]),)] = old_N + 1

            ### Case 2
            idx_increasing = (check == +1).nonzero(as_tuple=True)
            # index of the reactions that led to observed state increasing
            #if idx_increasing[0].size()[0]>0:
            #Next line is not working? Doing some cloning in between?
            ##new_param[idx_increasing][..., -1] -= 1
            new_param[idx_increasing+(torch.tensor([-1]),)] -=1

            ### Case 3
            idx_constant = (check == 0).nonzero(as_tuple=True)
            # index of the reactions that led to observed state staying constant. No jump update

        #Non-batched data:
        else:
            if check == 0:
                ## The observed reaction does not influence the estimated states, thetas stay the same. replace exact by obs
                new_param[...,-self.o_space.dimensions:] = observation
            elif check == 1:
                # one observed state increased. thetas stay the same, N decreases by one, replace exact states by observation
                #new_param[...,-1] -= 1
                new_param[...,-self.o_space.dimensions:] = observation

            elif check == -1:
                # one observed state decreased. moment matching to see how it affects the estimated state. more likely rate will increase the moment more

                idx_tuple = (observation - exact_states).nonzero(as_tuple=True)
                idx = idx_tuple[-1]
                # getting the outgoing rates from y_idx
                c = self.t_model.c[action,param.shape[-1]-self.o_space.dimensions+idx,:param.shape[-1]-self.o_space.dimensions].squeeze()
                d = self.t_model.d[action,param.shape[-1]-self.o_space.dimensions+idx]
                # Calculate new thetas
                #update increases the ratio of the rate increasing the estimate divided by the total rate of leaving y.
                new_theta = theta + (c)/(c.sum()+d)
                new_param = torch.cat((new_theta,observation))

            else:
                raise NotImplementedError


        return new_param

    def log_prob(self, state: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        exact_states_param = param[..., -self.o_space.dimensions:].clone()
        exact_states_state = state[...,-self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()

        estimated_states = state[...,:-self.o_space.dimensions]
        log_prob = torch.zeros(state.shape[:-1]).log()
        # prob is zero for all states, that do not have the same observed states as the parameter
        # Compute the log prob for the ones that have the same observed states:
        check = (exact_states_state==exact_states_param).prod(-1)==1
        log_prob[check] = torch.distributions.Poisson(theta).log_prob(estimated_states[check]).sum(-1)
        return log_prob

    def sample(self, param, num_samples=1):
        exact_states = param[..., -self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()
        states = torch.distributions.Poisson(theta).sample((num_samples,))
        return torch.cat((states,exact_states * torch.ones(*states.shape[:-1],exact_states.shape[-1])),-1)
    def moments(self, param: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        ## variance only contains the diagonal elements here.
        exact_states = param[...,-self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()

        return theta, theta, None


class TruncPoissonMonoCRNFilter(ContTimeProjectionFilter):
    o_model: ExactContTimeObservationModel
    t_model: TruncPoissonMonoCRNTransitionModel

    def __init__(self, t_model: PoissonMonoCRNTransitionModel, o_model: ExactContTimeObservationModel, initial_param: torch.Tensor,
                 initial_time: torch.Tensor, exp_method='MC', exp_samples=1000, device=None, sim_method=None,
                 sim_options=None, jump_optim=torch.optim.Adam, jump_opt_iter=500, jump_opt_options = {"lr":0.01}):
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
        #TODO batchwise

        #Split belief in estimate and exact_states
        exact_states = belief[...,-self.o_space.dimensions:].clone()
        theta = belief[...,:-self.o_space.dimensions].clone()

        unobs_dim = theta.shape[-1]
        c_xx = self.t_model.c[action,:unobs_dim,:unobs_dim]
        #dtheta_i/dt = sum(j) -c_ij theta_i + c_ji theta_j
        theta_drift = -c_xx.sum(-1)*theta + (c_xx * theta[...,None]).sum(-2)
        #batch x state_dim

        c_xy = self.t_model.c[action,:unobs_dim,unobs_dim:]
        r_x = c_xy.sum(-1)

        theta_drift += - (r_x+self.t_model.d[action,:unobs_dim]) * theta + self.t_model.b[action,:unobs_dim]
        # Concatenate with zeros for the exact states
        drift = torch.cat((theta_drift,torch.zeros(*theta_drift.shape[:-1],self.o_space.dimensions)),-1)
        return drift

    def jump_update(self, param, observation, action, **kwargs):
        #TODO batchwise
        batch_size = torch.broadcast_shapes(param.shape[:-1], observation.shape[:-1], action.shape[:-1])
        param = param.expand(*batch_size, param.shape[-1])
        action = action.expand(*batch_size,action.shape[-1])
        observation = observation.expand(*batch_size,observation.shape[-1])

        exact_states = param[...,-self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()
        new_param = param.clone()
        check = (observation - exact_states).sum(-1)
        #check dim: *batch_size
        # if check is -1, the unobserved states increased, while the observed decreased
        # if check is 1, the observed states increased, while the unobserved decreased
        # both lead to different jump updates
        # if check is 0, the reaction was from observed to observed or from unobserved to unobserved -> no update

        #Different handling for batched and non-batched data
        ## TODO: change the jump updates for batched data. From param contains N to param containst the exact states.
        #Batched data:
        if observation.dim() > 1:
            ##Case 1
            idx_decreasing = (check == -1).nonzero(as_tuple = True)
            #index of the reactions that led to observed state decreasing
            # observation[idx_decreasing], action[idx_decreasing], param[idx_decreasing] describe these reactions and have dim: (number_of_reactions x (obs/action/param)_dim

            obs_idx = observation[idx_decreasing].nonzero(as_tuple=True)[-1]
            #obs_idx describes which of the observed states decreased
            c = self.t_model.c[action[idx_decreasing].squeeze(), param.shape[-1] + obs_idx, :param.shape[-1]]
            # rates of the observed states to the unobserved, given the action. Dim: number_of_reactions x param_dim (unobserved states)

            old_N = param[idx_decreasing][...,-1]
            # Old buffer size parameter of multinomial dist for the indexed reactions. dim: number_of_reactions
            # adding a dimension for the param_dim with None to some elements
            new_param[idx_decreasing] = (old_N[...,None] * param[idx_decreasing] + c / c.sum(-1)[...,None]) / (old_N[...,None] + 1)
            # Last element is wrong, because param[-1] contains the N parameter. Is exchanged in next step.
            # Buffer size is increased by one for this type of reactions
            new_param[idx_decreasing+(torch.tensor([-1]),)] = old_N + 1

            ### Case 2
            idx_increasing = (check == +1).nonzero(as_tuple=True)
            # index of the reactions that led to observed state increasing
            #if idx_increasing[0].size()[0]>0:
            #Next line is not working? Doing some cloning in between?
            ##new_param[idx_increasing][..., -1] -= 1
            new_param[idx_increasing+(torch.tensor([-1]),)] -=1

            ### Case 3
            idx_constant = (check == 0).nonzero(as_tuple=True)
            # index of the reactions that led to observed state staying constant. No jump update

        #Non-batched data:
        else:
            if check == 0:
                ## The observed reaction does not influence the estimated states, thetas stay the same. replace exact by obs
                new_param[...,-self.o_space.dimensions:] = observation
            elif check == 1:
                # one observed state increased. thetas stay the same, N decreases by one, replace exact states by observation
                #new_param[...,-1] -= 1
                new_param[...,-self.o_space.dimensions:] = observation


            elif check == -1:
                # one observed state decreased. moment matching to see how it affects the estimated state. more likely rate will increase the moment more

                idx_tuple = (observation - exact_states).nonzero(as_tuple=True)
                idx = idx_tuple[-1]
                # getting the outgoing rates from y_idx
                c = self.t_model.c[action,param.shape[-1]-self.o_space.dimensions+idx,:param.shape[-1]-self.o_space.dimensions].squeeze()
                d = self.t_model.d[action,param.shape[-1]-self.o_space.dimensions+idx]
                # Calculate new thetas
                #update increases the ratio of the rate increasing the estimate divided by the total rate of leaving y.
                new_theta = theta + (c)/(c.sum()+d)
                new_param = torch.cat((new_theta,observation))

            else:
                raise NotImplementedError


        return new_param

    def log_prob(self, state: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        batch_size = torch.broadcast_shapes(state.shape[:-1],param.shape[:-1])
        if len(batch_size)>=2:
            state=state.expand(*batch_size,state.shape[-1])
        exact_states_param = param[..., -self.o_space.dimensions:].clone()
        exact_states_state = state[...,-self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()

        estimated_states = state[...,:-self.o_space.dimensions]
        log_prob = torch.zeros(state.shape[:-1]).log()
        if len(batch_size)>=2:
            check = (exact_states_state[0,0] == exact_states_param[0,0]).prod(-1) == 1
            log_prob[:,:,check] = torch.distributions.Poisson(theta).log_prob(estimated_states[:,:,check,:]).sum(-1)
        else:
            # prob is zero for all states, that do not have the same observed states as the parameter
            # Compute the log prob for the ones that have the same observed states:
            check = (exact_states_state==exact_states_param).prod(-1)==1
            log_prob[check] = torch.distributions.Poisson(theta).log_prob(estimated_states[check]).sum(-1)
        return log_prob

    def sample(self, param, num_samples=1):
        exact_states = param[..., -self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()
        states = torch.distributions.Poisson(theta).sample((num_samples,))
        return torch.cat((states,exact_states * torch.ones(*states.shape[:-1],exact_states.shape[-1])),-1).clip(max = self.t_model.s_space.cardinalities -1)
    def moments(self, param: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        ## variance only contains the diagonal elements here.
        exact_states = param[...,-self.o_space.dimensions:].clone()
        theta = param[...,:-self.o_space.dimensions].clone()

        return theta, theta, None