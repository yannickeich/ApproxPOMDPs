from packages.model.components.filter.filter import Filter
from packages.model.components.transition_model import TransitionModel
from packages.model.components.observation_model import ObservationModel, DiscreteTimeObservationModel
import torch
from abc import ABC, abstractmethod

EPS = 1e-6


class DiscreteTimeProjectionFilter(Filter, ABC):
    o_model: DiscreteTimeObservationModel

    def __init__(self, t_model: TransitionModel, o_model: ObservationModel, initial_param: torch.Tensor,
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

    @abstractmethod
    def operator_grad_log(self, state: torch.Tensor, action: torch.Tensor, param: torch.Tensor):
        """
        Computes the Markovian adjoint operator applied to the grad log of the prob.
        This is needed for the ODE that describes the parameters.
        :param state: state
        :param action: action
        :param param: parameter of the filtering distribution
        :return: part of the right hand side of the ODE describing the parameter evolution
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
        Updates the belief given an observation and finds the best param in KL sense that describe the new posterior.
        Learning rate and iterations are hyperparemeter that have to be carefully tuned for each parametric family.

        :param param: initial param
        :param observation: observation used for the update
        :param action: action used at that time
        :return: new parameter after update
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

        batch_size = torch.broadcast_shapes(param.shape[:-1], observation.shape[:-1], action.shape[:-1])

        unc_param = self.belief_to_unc_belief(param).expand(*batch_size, param.shape[-1]).detach().clone()
        unc_param.requires_grad_(True)

        optimizer = self.jump_optim([unc_param],**self.jump_opt_options)
        for i in range(self.jump_opt_iter):
            optimizer.zero_grad()

            def func(state):
                param_new = self.unc_belief_to_belief(unc_param)
                return self.o_model.log_prob(state, action[..., None, :],
                                             observation[..., None, :]).exp() * self.log_prob(state,
                                                                                              param_new[..., None, :])

            def closure():
                optimizer.zero_grad()
                objective = -self.expectation_state_func(func,param,method=method,num_samples=num_samples).sum()
                objective.backward()
                return objective


            objective = -self.expectation_state_func(func, param, method=method, num_samples=num_samples).sum()
            objective.backward()
            optimizer.step(closure)


        unc_param.requires_grad = False
        param_new = self.unc_belief_to_belief(unc_param)

        return param_new

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
        # If not called otherwise, use filters expectation method
        try:
            method = kwargs['method']
        except KeyError:
            method = self.exp_method
        try:
            num_samples = kwargs['num_samples']
        except KeyError:
            num_samples = self.exp_samples

        # dim before permute: action_batch x state_dim x belief_batch
        # output dim action_batch x belief_batch x state_dim
        def func2(state):
            return self.increment(state, action[..., None, :], belief[..., None, :], method=method).permute(-1, *range(
                len(state.shape[:-1])))

        def func(state):
            return self.operator_grad_log(state, action[..., None, :], belief[..., None, :]).permute(-1, *range(
                len(state.shape[:-1])))

        ## To compare. this one does the fisher matrix inside the expectation, if increment(including fisher_matrix) can be written in closed form, one should overwrite drift
        # drift = self.expectation_state_func(func2, belief, method=method).permute(*range(1, len(belief.shape[:-1]) + 1),0)

        # drift = (self.inv_fisher_matrix(belief, method=method, num_samples=num_samples) @
        #          self.expectation_state_func(func, belief, method=method, num_samples=num_samples).permute(
        #              *range(1, len(belief.shape[:-1]) + 1),
        #              0)[..., None]).sum(dim=-1)
        operator_grad_log = self.expectation_state_func(func, belief, method=method, num_samples=num_samples).permute(
            *range(1, len(belief.shape[:-1]) + 1),0)
        drift = torch.linalg.solve(self.fisher_matrix(belief,method=method) + lambda_ * torch.diag_embed(torch.ones_like(belief)),operator_grad_log[...,None]).sum(dim=-1)
        return drift

    def increment(self, state, action, param, **kwargs):
        """
        function inside the expectation for the increment of parameter ode

        :param state: state variable
        :param param: parameter vector of the filter
        :param action: action signal
        """

        return (self.inv_fisher_matrix(param, **kwargs) * self.operator_grad_log(state, action, param)[..., None]).sum(
            dim=-1)
