import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule
from wacky import functional as funky


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
            in_features=n_dims * in_features,
            out_features=n_dims * out_features,
            module=nn.Linear,
            activation=nn.Sigmoid()
        )

    def forward(self, x, incoming_connections):

        connect = self.connector(incoming_connections.reshape(1, -1))
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

        if units is not None:
            for u in units:
                self.append_encoded_weights_layer(u, hidden_activation)

    def append_encoded_weights_layer(self, units, activation=None):
        new_layer = SelfAssemblingEncodedWeights(
            in_features=self.out_features,
            out_features=units,
            n_dims=self.n_dims,
            shared_layer=self.shared_layer,
            activation = activation
        )
        self.out_features += units
        self.layers.append(new_layer)

    def learn(self, *args, **kwargs):
        for layer in self.layers:
            if funky.has_method(layer, 'learn'):
                layer.learn( *args, **kwargs)

    def forward(self, x):
        in_connect = F.one_hot(th.arange(self.in_features), self.in_features).float()
        in_connect = self.in_feature_encoder(in_connect)

        for layer in self.layers:
            _x, _in_connect = layer(x.float(), in_connect)
            x = th.cat([x,_x], -1)
            in_connect = th.cat([in_connect, _in_connect], 0)
        return x


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
    agent.train(env, 10_500)
    agent.test(env, 100)

if __name__ == '__main__':
    main()
