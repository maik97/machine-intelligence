import torch as th
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class ContinuousDistributionModule(nn.Module):

    def __init__(self, in_features, action_shape, activation_mu=None, activation_sigma=None):
        super(ContinuousDistributionModule, self).__init__()

        self.mu_layers = [nn.Linear(in_features, units) for units in action_shape]
        self.sigma_layers = [nn.Linear(in_features, units) for units in action_shape]

        self.activation_mu = F.sigmoid if activation_mu is None else activation_mu
        self.activation_sigma = F.tanh if activation_sigma is None else activation_sigma

    def forward(self, x, deterministic=False):
        mu = [self.activation_mu(mu_layer(x)) for mu_layer in self.mu_layers]
        sigma = [self.activation_sigma(sigma_layer(x)) for sigma_layer in self.sigma_layers]

        distribution = Normal(th.stack(mu), th.stack(sigma))
        action = distribution.rsample() if not deterministic else mu
        log_prob = distribution.log_prob(action)
        return action, log_prob


class DiscreteDistributionModule(nn.Module):

    def __init__(self, in_features, action_n, activation=None):
        super(DiscreteDistributionModule, self).__init__()

        self.layer = nn.Linear(in_features, action_n)
        self.activation = nn.Softmax() if activation is None else activation

    def forward(self, x, deterministic=False):
        x = self.activation(self.layer(x))
        distribution = Categorical(x)
        action = distribution.sample() if not deterministic else th.argmax(x)
        log_prob = distribution.log_prob(action)
        return action, log_prob