import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces

from wacky.networks.layered_networks import MultiLayerPerceptron
from wacky.networks.actor_critic_networks import ActorCriticNetwork
from wacky.functional.gym_space_decoder import decode_gym_space
from wacky.functional.distributions import make_distribution_network
from wacky.backend.error_messages import check_type, raise_type_error


def make_shared_net_for_actor_critic(observation_space, shared_net, activation_shared):
    if shared_net is None:
        shared_net_module = None
        in_features = int(decode_gym_space(observation_space, allowed_spaces=[spaces.Box])[0])

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


def make_actor_net(action_space, in_features, actor_net=None, activation_actor=th.nn.ReLU()):
    if actor_net is None:
        actor_net = [64, 64]

    elif isinstance(actor_net, int):
        actor_net = [actor_net]

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

    action_layer = make_distribution_network(in_features=actor_net_module.out_features, space=action_space)
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
        activation_shared=nn.ReLU(),
        activation_actor=nn.ReLU(),
        activation_critic=nn.ReLU(),
):
    in_features, shared_net_module = make_shared_net_for_actor_critic(observation_space, shared_net, activation_shared)
    actor_net_module = make_actor_net(action_space, in_features, actor_net, activation_actor)
    critic_net_module = make_critic_net(in_features, critic_net, activation_critic)

    return ActorCriticNetwork(actor_net_module, critic_net_module, shared_net_module)
