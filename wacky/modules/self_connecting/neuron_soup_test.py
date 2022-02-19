import math
import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule

class SoupLayer(WackyModule):

    def __init__(self, n_neurons, bias=False):
        super(SoupLayer, self).__init__()

        self.weight = nn.Parameter(th.empty((n_neurons, n_neurons)))
        if bias:
            self.bias = nn.Parameter(th.empty(n_neurons))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: th.Tensor) -> th.Tensor:
        return F.linear(input, self.weight, self.bias)



class NeuronSoup(WackyModule):

    def __init__(self, in_features, out_features, n_neurons=64):
        super(NeuronSoup, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.input_encoder = WackyLayer(
            in_features=in_features,
            out_features=n_neurons,
            module=nn.Linear,
            activation=nn.ReLU()
        )

        self.soup = SoupLayer(
            n_neurons=n_neurons,
        )

        self.output_decoder = WackyLayer(
            in_features=n_neurons,
            out_features=out_features,
            module=nn.Linear,
            activation=nn.ReLU()
        )

    def forward(self, x):
        h = self.input_encoder(x)
        for i in range(2):
            h = F.sigmoid(self.soup(h))
        print(h)
        return self.output_decoder(h)
