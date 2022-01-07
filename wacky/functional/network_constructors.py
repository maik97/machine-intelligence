import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces

from wacky.functional.gym_space_decoder import decode_gym_space


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
            raise NotImplemented()

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


class ActorCriticNetworkWithShared(nn.Module):

    def __init__(self, shared_net_module, actor_net_module, critic_net_module):
        super(ActorCriticNetworkWithShared, self).__init__()

        self.shared_net_module = shared_net_module
        self.actor_net_module = actor_net_module
        self.critic_net_module = critic_net_module

    def forward(self, x):
        x = self.shared_net_module(x)
        return self.actor_net_module(x), self.critic_net_module(x)


class ActorCriticNetworkNoShared(nn.Module):

    def __init__(self, actor_net_module, critic_net_module):
        super(ActorCriticNetworkNoShared, self).__init__()

        self.actor_net_module = actor_net_module
        self.critic_net_module = critic_net_module

    def forward(self, x):
        return self.actor_net_module(x), self.critic_net_module(x)


def actor_critic_net_arch(
        observation_space,
        activation_hidden=F.relu,
        activation_actor=F.tanh,
        activation_critic=None,
        shared_net_units=None,
        actor_net_units=None,
        critic_net_units=None,
):
    if actor_net_units is None:
        actor_net_units = [64, 64]

    if critic_net_units is None:
        critic_net_units = [64, 64]

    if shared_net_units is not None:
        if not isinstance(shared_net_units, list):
            raise TypeError("'shared_net_units' must be either type list or None, not", type(shared_net_units))
        else:
            shared_net_module = MultiLayerPerceptron(
                n_inputs=decode_gym_space(observation_space, allowed_spaces=[spaces.Box]),
                layer_units=shared_net_units,
                activation_hidden=activation_hidden,
                activation_out=activation_hidden
            )
            n_inputs = shared_net_module.out_features

    else:
        shared_net_module = None
        n_inputs = decode_gym_space(observation_space, allowed_spaces=[spaces.Box])

    actor_net_module = MultiLayerPerceptron(
        n_inputs=n_inputs,
        layer_units=actor_net_units,
        activation_hidden=activation_hidden,
        activation_out=activation_actor
    )

    critic_net_module = MultiLayerPerceptron(
        n_inputs=n_inputs,
        layer_units=critic_net_units,
        activation_hidden=activation_hidden,
        activation_out=activation_critic
    )

    if shared_net_module is not None:
        return ActorCriticNetworkWithShared(shared_net_module, actor_net_module, critic_net_module)
    else:
        return ActorCriticNetworkNoShared(actor_net_module, critic_net_module)
