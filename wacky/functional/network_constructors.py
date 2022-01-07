import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces

from wacky.functional.gym_space_decoder import decode_gym_space
from wacky.functional.distributions import make_action_distribution
from wacky.backend.error_messages import check_type, raise_type_error


class ParallelLayers(nn.Module):

    def __init__(self, in_features, out_features_list, activations_list=None, module=nn.Linear, *args, **kwargs):
        super(ParallelLayers, self).__init__()
        check_type(in_features, int, 'in_features')
        check_type(out_features_list, list, 'out_features_list')
        check_type(activations_list, list, 'activations_list', allow_none=True)
        check_type(module, nn.Module, 'module')

        self.in_features = in_features
        self.layers = [module(in_features, out_features, *args, **kwargs) for out_features in (out_features_list)]
        self.activations = [None for i in range(len(self))] if activations_list is None else activations_list

    @property
    def out_features(self):
        return [layer.out_feature for layer in self.layers]

    def __len__(self):
        return len(self.layers)

    def forward(self, x):
        out = []
        for i in range(len(self)):
            x_i = self.layers[i](x)
            if self.activations[i] is not None:
                x_i = self.activations[i](x_i)
            out.append(x_i)
        return out

    def append_layer(self, units, activation=None, module=nn.Linear, *args, **kwargs):
        self.layers.append(module(self.in_features, units,  *args, **kwargs))
        self.activations.append(activation)


class MultiLayerPerceptron(nn.Module):

    def __init__(
            self,
            in_features: int,
            layer_units: list = None,
            activation_hidden=None,
            activation_out=None
    ):

        super(MultiLayerPerceptron, self).__init__()
        check_type(in_features, int, 'in_features')
        check_type(layer_units, (list, int), 'layer_units')
        check_type(activation_hidden, list, 'activation_hidden', allow_none=True)
        check_type(activation_out, list, 'activation_out', allow_none=True)

        self.in_features = in_features
        self.layers = []

        if layer_units is not None:
            for i in range(len(layer_units) - 1):
                self.append_layer(layer_units[i + 1], activation_hidden)
            if isinstance(layer_units[-1], int):
                self.append_layer(layer_units[-1], activation_out)
            elif isinstance(layer_units[-1], list):
                self.layers.append(
                    ParallelLayers(
                        self.out_features,
                        layer_units[-1],
                        activations_list = [activation_out for i in range(len(layer_units[-1]))]
                    )
                )

    def __len__(self):
        return len(self.layers)

    @property
    def out_features(self):
        if len(self) == 0:
            return self.in_features
        else:
            return self.layers[-1].out_features

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def append_layer(self, units, activation, module: nn.Module = nn.Linear, *args, **kwargs):
        self.layer_list.append(module(self.out_features, units,  *args, **kwargs))
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
            in_features=decode_gym_space(observation_space, allowed_spaces=[spaces.Box]),
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
            in_features=in_features,
            layer_units=actor_net,
            activation_hidden=activation_actor,
            activation_out=activation_actor
        )
    elif isinstance(actor_net, nn.Module):
        actor_net_module = actor_net
    else:
        raise_type_error(actor_net, (None, list, nn.Module), 'actor_net')

    action_layer = make_action_distribution(in_features=actor_net_module.out_features, space=action_space)
    actor_net_module.layers.append(action_layer)

    return actor_net_module


def make_critic_net(in_features, critic_net, activation_critic):
    if critic_net is None:
        critic_net = [64, 64]

    if isinstance(critic_net, list):
        critic_net_module = MultiLayerPerceptron(
            in_features=in_features,
            layer_units=critic_net,
            activation_hidden=activation_critic,
            activation_out=activation_critic
        )
    elif isinstance(critic_net, nn.Module):
        critic_net_module = critic_net
    else:
        raise_type_error(critic_net, (None, list, nn.Module), 'critic_net')

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
