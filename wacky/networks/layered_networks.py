"""
class TemplateNetwork(nn.Module):

    def __init__(self, in_features, out_features, *args, **kwargs):
        super(NeuralNetwork, self).__init__()
        backend.check_type(in_features, int, 'in_features')
        backend.check_type(...)

        self.in_features = in_features
        self.out_features = out_features
        self.layers = []
        ...

    @property
    def in_features(self):
        # alternative to self.in_features

    @property
    def out_features(self):
        # alternative to self.out_features

    def forward(self, x):
        ...
        return out

    def append_layer(self, units, activation=None, module=nn.Linear, *args, **kwargs):
        self.layers.append(...)
"""
import torch.nn as nn
from wacky import backend


class ParallelLayers(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features_list: list,
            activations_list: list = None,
            module: nn.Module = nn.Linear,
            *args, **kwargs
    ):

        super(ParallelLayers, self).__init__()

        backend.check_type(in_features, int, 'in_features')
        backend.check_type(out_features_list, list, 'out_features_list')
        backend.check_type(activations_list, list, 'activations_list', allow_none=True)
        backend.check_type(module, nn.Module, 'module')

        self.in_features = in_features
        self.layers = [module(in_features, out_features, *args, **kwargs) for out_features in out_features_list]
        self.activations = [None] * len(self) if activations_list is None else activations_list

    @property
    def out_features(self):
        return [layer.out_features for layer in self.layers]

    def __len__(self):
        return len(self.layers)

    def forward(self, x):
        out = []
        for layer, activation in zip(self.layers, self.activations):
            out.append(layer(x)) if activation is None else out.append(activation(layer(x)))
        return out

    def append_layer(self, units, activation=None, module=nn.Linear, *args, **kwargs):
        backend.check_type(module, nn.Module, 'module')
        self.layers.append(module(self.in_features, units, *args, **kwargs))
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
        backend.check_type(in_features, int, 'in_features')
        backend.check_type(layer_units, (list, int), 'layer_units')
        backend.check_type(activation_hidden, list, 'activation_hidden', allow_none=True)
        backend.check_type(activation_out, list, 'activation_out', allow_none=True)

        self.in_features = in_features
        self.layers = []

        if layer_units is not None:

            for i in range(len(layer_units) - 1):
                self.append_layer(layer_units[i + 1], activation_hidden)

            if isinstance(layer_units[-1], int):
                self.append_layer(layer_units[-1], activation_out)

            elif isinstance(layer_units[-1], list):
                parallel_layers = ParallelLayers(
                    in_features=self.out_features,
                    out_features_list=layer_units[-1],
                    activations_list=[activation_out] * len(layer_units[-1])
                )
                self.layers.append(parallel_layers)

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

    def append_layer(self, units, activation, module=nn.Linear, *args, **kwargs):
        backend.check_type(module, nn.Module, 'module')
        self.layer_list.append(module(self.out_features, units, *args, **kwargs))
        if activation is not None:
            self.layer_list.append(activation)
