import torch as th
from torch import nn
from torch.distributions import Normal, Categorical



class ContinuousDistributionModule(nn.Module):

    def __init__(self, in_features, action_shape, activation_mu=None, activation_sigma=None):
        super(ContinuousDistributionModule, self).__init__()

        units = action_shape[0]
        self.mu_layer = nn.Linear(in_features, units)# for units in action_shape]
        self.sigma_layer = nn.Linear(in_features, units)# for units in action_shape]

        self.activation_mu = nn.Tanh() if activation_mu is None else activation_mu
        self.activation_sigma = nn.Sigmoid() if activation_sigma is None else activation_sigma

    def make_dist(self, x):
        mu = self.activation_mu(self.mu_layer(x))  # for mu_layer in self.mu_layers]
        sigma = self.activation_sigma(self.sigma_layer(x))  # for sigma_layer in self.sigma_layers]
        sigma = th.clamp(sigma, min=0.4)
        return Normal(mu, sigma), mu

    def forward(self, x, deterministic=False):
        distribution, mu = self.make_dist(x)
        action = distribution.rsample() if not deterministic else mu
        log_prob = distribution.log_prob(action)
        return action, log_prob

    def eval_action(self,x, action):
        distribution, _ = self.make_dist(x)
        log_prob = distribution.log_prob(action)
        return log_prob


class DiscreteDistributionModule(nn.Module):

    def __init__(self, in_features, action_n, activation=None):
        super(DiscreteDistributionModule, self).__init__()

        self.layer = nn.Linear(in_features, action_n)
        self.activation = nn.Softmax() if activation is None else activation

    def make_dist(self, x):
        x = self.activation(self.layer(x))
        return Categorical(x)

    def forward(self, x, deterministic=False):
        distribution = self.make_dist(x)
        action = distribution.sample() if not deterministic else th.argmax(x)
        log_prob = distribution.log_prob(action)
        return action, log_prob

    def eval_action(self,x, action):
        distribution = self.make_dist(x)
        log_prob = distribution.log_prob(action)
        return log_prob