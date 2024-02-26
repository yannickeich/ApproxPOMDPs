from abc import ABC, abstractmethod
from packages.model.components.spaces import Space, ContinuousSpace
import torch


class ObservationModel(ABC):
    def __init__(self, s_space: Space, a_space: Space, o_space: Space):
        """
        Base class for observation model
        :param s_space: state space
        :param a_space: action space
        :param o_space: observation space
        """
        self.s_space = s_space
        self.a_space = a_space
        self.o_space = o_space


class DiscreteTimeObservationModel(ObservationModel, ABC):
    @abstractmethod
    def log_prob(self, state: torch.Tensor, action: torch.Tensor, observation):
        # return emission log probability
        raise NotImplementedError

    @abstractmethod
    def sample(self, state: torch.tensor, action: torch.Tensor):
        # return observation sample
        raise NotImplementedError


class PoissonObservationModel(DiscreteTimeObservationModel, ABC):
    def __init__(self, s_space: Space, a_space: Space, o_space: Space, obs_rate: torch.Tensor):
        """
        Discrte time observation model with time points distributed according to a poisson process
        :param s_space: state space
        :param a_space: action space
        :param o_space: observation space
        :param obs_rate: rate parameter of the Poisson process
        """
        super().__init__(s_space, a_space, o_space)
        self.obs_rate = obs_rate

    def sample_times(self, t_span):
        """
        Samples possible observation times from Poisson distribution.
        Since all times are equally likely, we first sample number of observations and then choose random times from uniform dist.
        :param t_span: vector containing two elements: start and end time
        :return: t_obs: vector containing observation times
        """

        n_obs = int(torch.distributions.poisson.Poisson(self.obs_rate * (t_span[1] - t_span[0])).sample())
        t_obs = torch.sort(t_span[0] + torch.rand(n_obs) * (t_span[1] - t_span[0]))[0]
        return t_obs


class GaussianObservationModel(PoissonObservationModel):
    def __init__(self,s_space: Space, a_space:Space, o_space: Space, obs_rate:torch.Tensor, variance: torch.Tensor, linear:torch.Tensor):
        """
        Discrete time observation model with time points distributed according to a poisson process. The observations are sampled using a gaussian distribution with the Linear @ state as the mean.
        :param s_space:
        :param a_space:
        :param o_space:
        :param obs_rate:
        :param variance:
        """
        super().__init__(s_space,a_space,o_space,obs_rate)
        self.variance = variance
        self.linear = linear

    def sample(self,state: torch.tensor, action: torch.Tensor):

        obs = torch.distributions.multivariate_normal.MultivariateNormal(loc= (self.linear @ state[...,None].float()).squeeze(-1),covariance_matrix = self.variance).sample()

        return obs

    def log_prob(self,state:torch.Tensor, action:torch.Tensor,observation:torch.Tensor):

        return torch.distributions.multivariate_normal.MultivariateNormal(loc=(self.linear @ state[...,None].float()).squeeze(-1),covariance_matrix=self.variance).log_prob(observation)


class ExactContTimeObservationModel(ObservationModel):
    def __call__(self, state: torch.Tensor):
        # return observed states
        raise NotImplementedError