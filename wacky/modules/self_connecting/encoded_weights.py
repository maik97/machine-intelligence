import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule
from wacky import functional as funky

class ContinEncodedWeights(WackyModule):

    def __init__(self, in_features, out_features, n_dims=3, init_connections=None, shared_layer=None, activation=None):
        super(ContinEncodedWeights, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

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
            out_features=out_features,
            module=nn.Linear,
            #activation=nn.Tanh()
        )

        self.connections = th.rand(in_features, n_dims) if init_connections is None else init_connections
        self.update_weights(connected_neurons=self.connections)

    def update_weights(self, connected_neurons):
        w = self.calc_weight(connected_neurons)
        w = self.calc_weight_hidden(w)
        self.weight = th.transpose(w, 0, 1)

    def learn(self, *args, **kwargs):
        self.update_weights(connected_neurons=self.connections)

    def forward(self, x):
        x = F.linear(x, self.weight)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SimpleContinEncodedWeights(WackyModule):

    def __init__(self, in_features, out_features, n_dims=3, init_connections=None):
        super(SimpleContinEncodedWeights, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.calc_weight = WackyLayer(
            in_features=n_dims,
            out_features=out_features,
            module=nn.Linear,
        )

        self.connections = th.rand(in_features, n_dims) if init_connections is None else init_connections
        self.update_weights(connected_neurons=self.connections)

    def update_weights(self, connected_neurons):
        w = self.calc_weight(connected_neurons)
        self.weight = th.transpose(w, 0, 1)

    def learn(self, *args, **kwargs):
        self.update_weights(connected_neurons=self.connections)

    def forward(self, x):
        return F.linear(x, self.weight)


class OneHotEncodedWeightsLayer(WackyModule):

    def __init__(self, in_features, out_features):
        super(OneHotEncodedWeightsLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.calc_weight = WackyLayer(
            in_features=in_features,
            out_features=out_features,
            module=nn.Linear,
            activation=nn.Tanh()
        )

        self.update_weights()

    def update_weights(self):
        w = F.one_hot(th.arange(self.in_features), self.in_features).type(th.float)
        w = self.calc_weight(w)
        self.weight = th.transpose(w, 0, 1)

    def learn(self, *args, **kwargs):
        self.update_weights()

    def forward(self, x):
        return F.linear(x, self.weight)


class SlowOneHotEncodedWeightsLayer(WackyModule):

    def __init__(self, in_features, out_features):
        super(SlowOneHotEncodedWeightsLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.calc_weight = WackyLayer(
            in_features=in_features + out_features,
            out_features=1,
            module=nn.Linear,
            activation=nn.Tanh()
        )

        self.update_weights()

    def update_weights(self):
        weight = []
        for i in range(self.out_features):
            w_i = []
            for j in range(self.in_features):
                w_i.append(self.weight_between(i, j))
            weight.append(th.stack(w_i).reshape(self.in_features))
        self.weight = th.stack(weight).reshape(self.out_features, self.in_features)

    def weight_between(self, a, b):
        assert isinstance(a, int) and isinstance(b, int)
        a = F.one_hot(th.tensor(a), self.out_features)
        b = F.one_hot(th.tensor(b), self.in_features)
        x = th.cat([a, b]).reshape(1, -1).type(th.float)
        return self.calc_weight(x)

    def learn(self, *args, **kwargs):
        self.update_weights()

    def forward(self, x):
        return F.linear(x, self.weight)


class SharedEncodedWeightsNetwork(WackyModule):

    def __init__(self, in_features, units=None, shared_layer_units=16, n_dims=3, hidden_activation=None):
        super(SharedEncodedWeightsNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.in_features = in_features
        self.out_features = in_features
        self.n_dims = n_dims

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
        new_layer = ContinEncodedWeights(
            in_features=self.out_features,
            out_features=units,
            n_dims=self.n_dims,
            shared_layer=self.shared_layer,
            activation = activation
        )
        self.out_features = units
        self.layers.append(new_layer)

    def learn(self, *args, **kwargs):
        for layer in self.layers:
            if funky.has_method(layer, 'learn'):
                layer.learn( *args, **kwargs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x.float())
        return x
