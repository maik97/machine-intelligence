import math
import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule
from wacky import functional as funky


class VariableSizeLayer(WackyModule):

    def __init__(self, max_in_features=64, max_out_features=16, n_dims=3, activation=None,  shared_net=None):
        super(VariableSizeLayer, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.n_dim = n_dims
        self.activation = activation

        if shared_net is not None:
            self.hidden_net = shared_net
        else:
            self.hidden_net = self.make_hidden_net(self.n_dim, 16)

        self.weight_net = WackyLayer(
            in_features=shared_net.out_features,
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

    def calc_weights(self, in_features, out_features=None):
        one_hot_ins = F.one_hot(th.arange(in_features), self.max_in_features).float()

        out_features = out_features if out_features is not None else self.max_out_features
        one_hot_outs = th.arange(out_features).reshape(-1, 1).repeat(1, 8)
        one_hot_outs = F.one_hot(one_hot_outs, self.max_out_features).float()

        weights = []
        for out in one_hot_outs:
            embedded = self.embedded_weight_space(th.cat([one_hot_ins, out], -1))
            w = self.hidden_net(embedded)
            w = self.weight_net(w)
            weights.append(w)

        return th.stack(weights).reshape(out_features, in_features)

    def forward(self, input: th.Tensor, out_features=None) -> th.Tensor:
        batch_size, in_features = input.size()
        return F.linear(input, self.calc_weights(in_features, out_features))


class SelfAssemblingEncodedWeights(WackyModule):

    def __init__(
            self,
            in_features,
            out_features,
            n_dims=3,
            shared_layer=None,
            activation=None
    ):
        super(SelfAssemblingEncodedWeights, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.n_dims = n_dims

        if shared_layer is None:
            self.calc_weight = WackyLayer(
                in_features=n_dims,
                out_features=16,
                module=nn.Linear,
                activation=nn.Tanh()
            )
        else:
            self.calc_weight = shared_layer

        self.calc_weight_hidden = WackyLayer(
            in_features=self.calc_weight.out_features,
            out_features=in_features,
            module=nn.Linear,
        )

        self.connector = WackyLayer(
            in_features=n_dims,
            out_features=n_dims * out_features,
            module=nn.Linear,
            activation=nn.Sigmoid()
        )

        self.importance = WackyLayer(
            in_features=n_dims,
            out_features=1,
            module=nn.Linear,
        )

    def forward(self, x, incoming_connections):
        print(x)
        print(incoming_connections)
        print()
        important = self.importance(incoming_connections)
        print(important)
        important_relu = F.relu(important)
        print(important_relu)
        important_softmax = F.softmax(important_relu, dim=0)
        print(important_softmax)
        important_mask = (important_relu > 0.0).float()
        print(important_mask)

        incoming_connections = incoming_connections * important_mask
        x = x * important_softmax
        print()
        print(x)
        print(incoming_connections)
        exit()

        connect = self.connector(incoming_connections)
        connect = connect.reshape(-1, self.n_dims)

        weight = self.calc_weight(connect)
        weight = self.calc_weight_hidden(weight)

        x = F.linear(x, weight)

        if self.activation is not None:
            x = self.activation(x)
        return x, connect


class SharedEncodedWeightsNetwork(WackyModule):

    def __init__(self, in_features, units=None, shared_layer_units=16, n_dims=3, hidden_activation=None):
        super(SharedEncodedWeightsNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.in_features = in_features
        self.out_features = in_features
        self._out_features = in_features
        self.n_dims = n_dims

        self.in_feature_encoder = WackyLayer(
            in_features=in_features,
            out_features=n_dims,
            module=nn.Linear,
            activation=nn.Sigmoid()
        )

        self.shared_layer = WackyLayer(
            in_features=n_dims,
            out_features=shared_layer_units,
            module=nn.Linear,
            activation=nn.Tanh()
        )

        self.test = VariableSizeLayer(shared_net=self.shared_layer)
        exit()

        if units is not None:
            for u in units:
                self.append_encoded_weights_layer(u, hidden_activation)

    def append_encoded_weights_layer(self, units, activation=None):
        new_layer = SelfAssemblingEncodedWeights(
            in_features=self._out_features,
            out_features=units,
            n_dims=self.n_dims,
            shared_layer=self.shared_layer,
            activation = activation
        )
        self._out_features += units
        self.out_features = units
        self.layers.append(new_layer)

    def learn(self, *args, **kwargs):
        for layer in self.layers:
            if funky.has_method(layer, 'learn'):
                layer.learn( *args, **kwargs)

    def forward(self, x):
        in_connect = F.one_hot(th.arange(self.in_features), self.in_features).float()
        in_connect = self.in_feature_encoder(in_connect)
        x = x.reshape(-1, 1)

        for layer in self.layers:
            _x, _in_connect = layer(x.float(), in_connect)
            x = th.cat([x,_x], 0)
            in_connect = th.cat([in_connect, _in_connect], 0)
        return _x


def main():
    import gym
    from wacky.networks import MultiLayerPerceptron, ActorNetwork
    from wacky.optimizer import TorchOptimizer
    from wacky.agents import REINFORCE
    from wacky.scores import MonteCarloReturns

    env = gym.make('CartPole-v0')

    in_features = funky.decode_gym_space(env.observation_space)[0]
    network = MultiLayerPerceptron(in_features=in_features)
    #network.append_layer(64, nn.ReLU(), SharedEncodedWeightsNetwork)
    network.append_layer([64, 64], None, SharedEncodedWeightsNetwork, hidden_activation=nn.ReLU())

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
