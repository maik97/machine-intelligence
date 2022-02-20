import math
import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule
from wacky import functional as funky


class VariableSizeLayer(WackyModule):

    def __init__(self, max_in_features=64, max_out_features=16, n_dims=3, activation=None, shared_net=None, l_id=0):
        super(VariableSizeLayer, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.in_features = max_in_features
        self.out_features = max_out_features
        self.n_dim = n_dims
        self.activation = activation
        self.l_id = l_id

        if shared_net is not None:
            self.hidden_net = shared_net
        else:
            self.hidden_net = self.make_hidden_net(self.n_dim, 16)

        self.weight_net = WackyLayer(
            in_features=self.hidden_net.out_features,
            out_features=1,
            module=nn.Linear,
        )

        self.embedded_weight_space = WackyLayer(
            in_features=max_in_features + max_out_features,
            out_features=n_dims,
            module=nn.Linear,
            activation=nn.Sigmoid()
        )

    def make_hidden_net(self, n_dims, hidden_features):
        return WackyLayer(
            in_features=n_dims,
            out_features=hidden_features,
            module=nn.Linear,
            activation=nn.Tanh()
        )

    def calc_weights_1(self, in_features, out_features=None):
        one_hot_ins = F.one_hot(th.arange(in_features), self.max_in_features).float()

        out_features = out_features if out_features is not None else self.max_out_features
        one_hot_outs = th.arange(out_features).reshape(-1, 1).repeat(1, in_features)
        one_hot_outs = F.one_hot(one_hot_outs, self.max_out_features).float()

        weights = []
        for out in one_hot_outs:
            embedded = self.embedded_weight_space(th.cat([one_hot_ins, out], -1))
            w = self.hidden_net(embedded)
            w = self.weight_net(w)
            weights.append(w)

        return th.stack(weights).reshape(out_features, in_features)

    def calc_weights(self, in_features, out_features=None):
        out_features = out_features if out_features is not None else self.max_out_features

        weights = []
        ins = th.arange(in_features) / in_features
        for i in range(out_features):
            embedded = [
                ins,
                th.tensor(i / out_features).repeat(in_features).float(),
                th.tensor(self.l_id).repeat(in_features).float(),
            ]
            embedded = th.cat(embedded).reshape(3, -1).transpose(0, 1)
            w = self.hidden_net(embedded)
            w = self.weight_net(w)
            weights.append(w.squeeze())

        return th.stack(weights).reshape(out_features, in_features)

    def forward(self, input: th.Tensor, out_features=None) -> th.Tensor:
        batch_size, in_features = input.size()
        return F.linear(input, self.calc_weights(in_features, out_features))


def main():
    import gym
    from wacky.networks import MultiLayerPerceptron, ActorNetwork
    from wacky.optimizer import TorchOptimizer
    from wacky.agents import REINFORCE
    from wacky.scores import MonteCarloReturns

    env = gym.make('CartPole-v0')

    in_features = funky.decode_gym_space(env.observation_space)[0]
    network = MultiLayerPerceptron(in_features=in_features)

    shared_net = WackyLayer(
        in_features=3,
        out_features=16,
        module=nn.Linear,
        activation=nn.Tanh()
    )
    network.append_layer(64, nn.ReLU(), VariableSizeLayer, shared_net=shared_net, l_id=0)
    #network.append_layer(64, nn.ReLU(), VariableSizeLayer, shared_net=shared_net, l_id=1)
    #network.append_layer([64, 64], None, SharedEncodedWeightsNetwork, hidden_activation=nn.ReLU())

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
    agent.train(env, 2_000)
    agent.test(env, 100)


if __name__ == '__main__':
    main()