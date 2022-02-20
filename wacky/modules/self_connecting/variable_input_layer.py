import math
import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule, SharedWeightEncoder
from wacky import functional as funky


class VariableInputLayer(WackyModule):

    def __init__(self, max_in_features=64, out_features=16, n_dims=3, activation=None, shared_net=None):
        super(VariableInputLayer, self).__init__()

        self.max_in_features = max_in_features
        self.in_features = max_in_features
        self.out_features = out_features
        self.n_dim = n_dims
        self.activation = activation

        if shared_net is not None:
            self.hidden_net = shared_net
        else:
            self.hidden_net = SharedWeightEncoder(n_dims=n_dims, max_one_hot=max_in_features)

        self.weight_net = WackyLayer(
            in_features=self.hidden_net.out_features,
            out_features=out_features,
            module=nn.Linear,
        )

    def calc_weights(self, in_features):
        w = self.hidden_net(in_features)
        w = self.weight_net(w).transpose(0, 1)
        return w

    def forward(self, input: th.Tensor) -> th.Tensor:
        batch_size, in_features = input.size()
        return F.linear(input, self.calc_weights(in_features))


def main():
    import gym
    from wacky.networks import MultiLayerPerceptron, ActorNetwork
    from wacky.optimizer import TorchOptimizer
    from wacky.agents import REINFORCE
    from wacky.scores import MonteCarloReturns

    env = gym.make('CartPole-v0')

    in_features = funky.decode_gym_space(env.observation_space)[0]
    network = MultiLayerPerceptron(in_features=in_features)

    shared_net = SharedWeightEncoder()
    #network.append_layer(64, nn.ReLU(), nn.Linear)
    #network.append_layer(64, nn.ReLU(), nn.Linear)
    network.append_layer(64, nn.ReLU(), VariableInputLayer, shared_net=shared_net)
    network.append_layer(64, nn.ReLU(), VariableInputLayer, shared_net=shared_net)

    network = ActorNetwork(
        action_space=env.action_space,
        network=network,
    )

    print('\n', network)
    optimizer = TorchOptimizer(
        optimizer='Adam',
        network_parameter=network,
        lr=0.001,
    )

    agent = REINFORCE(network, optimizer, returns_calc=MonteCarloReturns())
    agent.train(env, 10_000)
    agent.test(env, 100)


if __name__ == '__main__':
    main()