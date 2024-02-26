import torch
from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    @abstractmethod
    def sample(self, num_samples: int):
        raise NotImplementedError


class UniformSampler(Sampler):
    def __init__(self, bounds: torch.Tensor):
        self.bounds = bounds
        super().__init__(dimensions=bounds.shape[0])

    def sample(self, num_samples: int):
        return torch.distributions.Uniform(self.bounds[:, 0], self.bounds[:, 1]).sample((num_samples,))


class DiscreteUniformSampler(Sampler):
    def __init__(self,bounds:torch.Tensor):
        self.bounds = bounds
        super().__init__(dimensions = bounds.shape[0])

    def sample(self, num_samples: int):
        return torch.stack([torch.randint(self.bounds[i,0],self.bounds[i,1],(num_samples,)) for i in range(self.dimensions)],dim=-1)

class DirichletSampler(Sampler):
    def __init__(self, concentration: torch.Tensor):
        self.concentration = concentration
        super().__init__(dimensions=concentration.numel())

    def sample(self, num_samples: int):
        return torch.distributions.Dirichlet(concentration=self.concentration).sample((num_samples,))


class ExponentialSampler(Sampler):
    def __init__(self,rates:torch.Tensor):
        self.rates = rates
        super().__init__(dimensions=rates.shape[0])

    def sample(self, num_samples: int):
        return torch.distributions.Exponential(rate=self.rates).sample((num_samples,))


class MultinomialSampler(Sampler):
    def __init__(self,total_count:int,state_dim:int):
        self.state_dim = state_dim
        self.total_count = total_count

        super().__init__(dimensions=state_dim)

    def sample(self,num_samples: int):
        samples = torch.rand(num_samples,self.state_dim)
        samples /= samples.sum(-1)[:, None]
        samples *= self.total_count

        samples = samples.int()
        samples[:,-1] = self.total_count - samples[:,:-1].sum(-1)

        return samples