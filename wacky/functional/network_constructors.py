import torch as th
import torch.nn as nn
import torch.nn.functional as F

from wacky.backend.gym_space_decoder import decode_gym_space

def append_layer(layer_list, inputs, outputs, activation, th_nn=nn.Linear):
    layer_list.append(th_nn(inputs, outputs))
    if not activation is None:
        layer_list.append(activation)
    return layer_list

class MultiLayerPerceptron(nn.Module):
    
    def __init__(
            self,
            units_list: list,
            activation_hidden=None,
            activation_out=None
    ):
        """
        :type units_list: list
            First element is the number of input units, second to next-to-last are the numbers hidden neurons,
            last element is the number of output neurons.
        """
        super(MultiLayerPerceptron, self).__init__()
        
        if not isinstance(units_list, list):
            raise TypeError("'units_list' must be type list, not", type(units_list))
        elif len(units_list) < 2:
            raise TypeError("'units_list' must have at least two values (input, output layer)")
        else:
            self.layers = []
            for i in range(len(units_list)-2):
                append_layer(self.layers, units_list[i], units_list[i+1], activation_hidden)

            if isinstance(units_list[-1], int):
                append_layer(self.layers, units_list[-2], units_list[-1], activation_out)
            elif isinstance(units_list[-1], list):
                pass # TODO: multi output model
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
        action_space,
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
    
    if not shared_net_units is None:
        if not isinstance(shared_net_units, list):
            raise TypeError("'shared_net_units' must be either type list or None, not", type(shared_net_units))
        else:
            shared_net_units.insert(0, decode_gym_space(observation_space, allowed_spaces=['Discrete', 'Box']))
            shared_net_module = MultiLayerPerceptron(
                units_list=shared_net_units,
                activation_hidden=activation_hidden,
                activation_out=activation_hidden
            )
            actor_net_units.insert(0, shared_net_units[-1])
            critic_net_units.insert(0, shared_net_units[-1])

    else:
        num_inputs = decode_gym_space(observation_space, allowed_spaces=['Discrete', 'Box'])
        actor_net_units.insert(0, num_inputs)
        critic_net_units.insert(0, num_inputs)

    actor_net_units.append(decode_gym_space(action_space, allowed_spaces=['Discrete', 'Box']))
    critic_net_units.append(1)
    actor_net_module = MultiLayerPerceptron(
        units_list=actor_net_units,
        activation_hidden=activation_hidden,
        activation_out=activation_actor
    )

    critic_net_module = MultiLayerPerceptron(
        units_list=actor_net_units,
        activation_hidden=activation_hidden,
        activation_out=activation_critic
    )

    if not shared_net_units is None:
        return ActorCriticNetworkWithShared(shared_net_module, actor_net_module, critic_net_module)
    else:
        return ActorCriticNetworkNoShared(actor_net_module, critic_net_module)
