import math
import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule, SharedWeightEncoder
from wacky import functional as funky


class VariableInFeaturesLayer(WackyModule):

    def __init__(self, max_in_features=64, out_features=16, n_dims=3, activation=None, shared_net=None):
        super(VariableInFeaturesLayer, self).__init__()

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
        y = F.linear(input, self.calc_weights(in_features))
        if self.activation is not None:
            y = self.activation(y)
        return y


class VariableOutFeaturesLayer(WackyModule):

    def __init__(self, in_features, max_out_features=16, n_dims=3, activation=None, shared_net=None):
        super(VariableOutFeaturesLayer, self).__init__()

        self.in_features = in_features
        self.out_features = max_out_features
        self.max_out_features = max_out_features
        self.n_dim = n_dims
        self.activation = activation

        if shared_net is not None:
            self.hidden_net = shared_net
        else:
            self.hidden_net = SharedWeightEncoder(n_dims=n_dims, max_one_hot=max_out_features)

        self.weight_net = WackyLayer(
            in_features=self.hidden_net.out_features,
            out_features=in_features,
            module=nn.Linear,
        )

    def calc_weights(self, out_features):
        out_features = self.max_out_features if out_features is None else out_features
        w = self.hidden_net(out_features)
        w = self.weight_net(w)
        return w

    def forward(self, input: th.Tensor, out_features=None) -> th.Tensor:
        y = F.linear(input, self.calc_weights(out_features))
        if self.activation is not None:
            y = self.activation(y)
        return y


class VariableSizeLayer(WackyModule):

    def __init__(
            self,
            max_in_features,
            max_out_features,
            hidden_features=16,
            n_dims=3,
            activation_in_features=None,
            activation_out_features=None,
            shared_net=None
    ):

        super(VariableSizeLayer, self).__init__()

        self.in_features = max_in_features
        self.out_features = max_out_features

        if shared_net is not None:
            hidden_net = shared_net
        else:
            hidden_net = SharedWeightEncoder(n_dims=n_dims, max_one_hot=max_out_features)

        hidden_net.append(32)

        self.in_features_net = VariableInFeaturesLayer(
            max_in_features=max_in_features,
            out_features=hidden_features,
            n_dims=n_dims,
            activation=activation_in_features,
            shared_net=hidden_net
        )

        self.out_features_net = VariableOutFeaturesLayer(
            in_features=hidden_features,
            max_out_features=max_out_features,
            n_dims=n_dims,
            activation=activation_out_features,
            shared_net=hidden_net
        )

    def forward(self, input: th.Tensor, out_features=None) -> th.Tensor:
        return self.out_features_net(self.in_features_net(input), out_features)


class VariableSizeNetwork(WackyModule):

    def __init__(
            self,
            in_features,
            out_features,
            n_layer=2,
            max_features=128,
            hidden_features=16,
            n_dims=3,
            activation_hidden=None,
            shared_net=None
    ):
        super(VariableSizeNetwork, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layer
        self.max_features = max_features

        if shared_net is not None:
            self.hidden_net = shared_net
        else:
            self.hidden_net = SharedWeightEncoder(n_dims=n_dims, max_one_hot=max_features)
        self.hidden_net.append(32)

        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            l = VariableSizeLayer(
                max_features,
                max_features,
                hidden_features=hidden_features,
                n_dims=n_dims,
                activation_in_features=activation_hidden,
                activation_out_features=activation_hidden,
                shared_net=self.hidden_net
            )
            self.layers.append(l)

        self.size_net = WackyLayer(self.n_layers, 32)
        self.size_hidden = WackyLayer(32, 32)
        self.size_out = WackyLayer(32, 1)

        self.learn()

    def learn(self, *args, **kwargs):
        one_hot_ins = F.one_hot(th.arange(self.n_layers), self.n_layers).float()
        s = self.size_net(one_hot_ins)
        s = self.size_hidden(s)
        s = self.size_out(s)
        s = F.sigmoid(s)
        print(s)
        self.sizes = s * self.max_features
        print(self.sizes)

    def forward(self, x):
        for l, s in zip(self.layers, self.sizes):
            print(s)
            x = l(x, s)
        return x



def main():
    import gym
    from wacky.networks import MultiLayerPerceptron, ActorNetwork
    from wacky.optimizer import TorchOptimizer
    from wacky.agents import REINFORCE
    from wacky.scores import MonteCarloReturns

    # = gym.make('CartPole-v0')
    env = gym.make('LunarLanderContinuous-v2')

    in_features = funky.decode_gym_space(env.observation_space)[0]
    network = MultiLayerPerceptron(in_features=in_features)

    shared_net = SharedWeightEncoder()
    #network.append_layer(64, nn.ReLU(), nn.Linear)
    #network.append_layer(64, nn.ReLU(), nn.Linear)
    network.append_layer(64, nn.ReLU(), VariableSizeLayer, shared_net=shared_net)
    #network.append_layer(64, nn.ReLU(), VariableInFeaturesLayer, shared_net=shared_net)
    #network.append_layer(64, nn.ReLU(), VariableOutFeaturesLayer, shared_net=shared_net)

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