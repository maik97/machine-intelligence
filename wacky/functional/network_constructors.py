import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces

from wacky.functional.gym_space_decoder import decode_gym_space
from wacky.functional.distributions import make_action_distribution


class MultiLayerPerceptron(nn.Module):

    def __init__(
            self,
            n_inputs: int,
            layer_units: list,
            activation_hidden=None,
            activation_out=None
    ):

        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(layer_units, (list, int)):
            raise TypeError("'layer_units' must be type list, not", type(layer_units))

        self.n_inputs = n_inputs

        for i in range(len(layer_units) - 1):
            self.append_layer(layer_units[i + 1], activation_hidden)

        if isinstance(layer_units[-1], int):
            self.append_layer(layer_units[-1], activation_out)
        elif isinstance(layer_units[-1], list):
            # TODO: multi output model
            raise NotImplemented

    @property
    def in_features(self):
        return self.n_inputs

    @property
    def out_features(self):
        return self.layers[-1].out_features

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def append_layer(self, units, activation, th_nn=nn.Linear):
        if len(self.layers) == 0:
            inputs = self.n_input
        else:
            inputs = self.layers[-1].out_features

        self.layer_list.append(th_nn(inputs, units))
        if activation is not None:
            self.layer_list.append(activation)


class ActorCriticNetwork(nn.Module):

    def __init__(self, actor_net_module, critic_net_module, shared_net_module=None):
        super(ActorCriticNetwork, self).__init__()

        self.shared_net_module = shared_net_module
        self.actor_net_module = actor_net_module
        self.critic_net_module = critic_net_module

    def forward(self, x):
        if self.shared_net_module is not None:
            x = self.shared_net_module(x)
        return self.actor_net_module(x), self.critic_net_module(x)


def make_shared_net_for_actor_critic(observation_space, shared_net, activation_shared):
    if shared_net is None:
        shared_net_module = None
        in_features = decode_gym_space(observation_space, allowed_spaces=[spaces.Box])

    elif isinstance(shared_net, list):
        shared_net_module = MultiLayerPerceptron(
            n_inputs=decode_gym_space(observation_space, allowed_spaces=[spaces.Box]),
            layer_units=shared_net,
            activation_hidden=activation_shared,
            activation_out=activation_shared
        )
        in_features = shared_net_module.out_features

    elif isinstance(shared_net, nn.Module):
        shared_net_module = shared_net
        in_features = shared_net_module.out_features

    else:
        raise TypeError("'shared_net' type must be either [None, list, nn.Module], not", type(shared_net))

    return in_features, shared_net_module


def make_actor_net(action_space, in_features, actor_net, activation_actor):
    if actor_net is None:
        actor_net = [64, 64]

    if isinstance(actor_net, list):
        actor_net_module = MultiLayerPerceptron(
            n_inputs=in_features,
            layer_units=actor_net,
            activation_hidden=activation_actor,
            activation_out=activation_actor
        )
    elif isinstance(actor_net, nn.Module):
        actor_net_module = actor_net
    else:
        raise TypeError("'actor_net' type must be either [None, list, nn.Module], not", type(actor_net))

    action_layer = make_action_distribution(in_features=actor_net_module.out_features, space=action_space)
    actor_net_module.layers.append(action_layer)

    return actor_net_module


def make_critic_net(in_features, critic_net, activation_critic):
    if critic_net is None:
        critic_net = [64, 64]

    if isinstance(critic_net, list):
        critic_net_module = MultiLayerPerceptron(
            n_inputs=in_features,
            layer_units=critic_net,
            activation_hidden=activation_critic,
            activation_out=activation_critic
        )
    elif isinstance(critic_net, nn.Module):
        critic_net_module = critic_net
    else:
        raise TypeError("'critic_net' type must be either [None, list, nn.Module], not", type(critic_net))

    critic_net_module.append_layer(1, activation=None)

    return critic_net_module


def actor_critic_net_arch(
        observation_space,
        action_space,
        shared_net=None,
        actor_net=None,
        critic_net=None,
        activation_shared=F.relu,
        activation_actor=F.tanh,
        activation_critic=None,
):
    in_features, shared_net_module = make_shared_net_for_actor_critic(observation_space, shared_net, activation_shared)
    actor_net_module = make_actor_net(action_space, in_features, actor_net, activation_actor)
    critic_net_module = make_critic_net(in_features, critic_net, activation_critic)

    return ActorCriticNetwork(actor_net_module, critic_net_module, shared_net_module)
