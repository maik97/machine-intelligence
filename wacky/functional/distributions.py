import numpy as np
from gym import spaces
from wacky.functional.gym_space_decoder import decode_gym_space

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


def make_action_distribution(in_features, space: spaces.Space):

    # space.Dict, space.Tuple
    if isinstance(space, (spaces.Tuple, spaces.Dict)):
        return [make_action_distribution(in_features, subspace) for subspace in space]

    allowed_spaces = [
        spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.Tuple, spaces.Dict,# spaces.MultiBinary
    ]
    action_shape = decode_gym_space(space, allowed_spaces=allowed_spaces)

    # spaces.MultiBinary - not implemented!
    if isinstance(space, spaces.MultiBinary):
        if isinstance(action_shape, tuple):
            pass
        elif isinstance(action_shape, int):
            pass
        else:
            pass
    # space.Box
    elif isinstance(action_shape, tuple):
        return ContinuousDistributionModule(in_features, action_shape)
    # space.Discrete
    elif isinstance(action_shape, int):
        return DiscreteDistributionModule(in_features, action_shape)
    # space.MultiDiscrete
    elif isinstance(action_shape, np.ndarray):
        return [DiscreteDistributionModule(in_features, int(n)) for n in action_shape]
