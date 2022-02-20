import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule


class SharedWeightEncoder(WackyModule):

    def __init__(self, n_dims=3, out_features=16, max_one_hot=64, activation=nn.Tanh()):
        super(SharedWeightEncoder, self).__init__()

        self.n_dims = n_dims
        self.out_features = out_features
        self.max_one_hot = max_one_hot
        self.activation = activation

        self.embedded_weight_space = WackyLayer(
            in_features=max_one_hot,
            out_features=n_dims,
            module=nn.Linear,
            activation=nn.Sigmoid()
        )

        self.hidden_net = WackyLayer(
            in_features=n_dims,
            out_features=out_features,
            module=nn.Linear,
            activation=activation
        )

    def one_hot(self, int_val):
        one_hot_ins = F.one_hot(th.arange(int_val), self.max_one_hot).float()
        return self.embedded_weight_space(one_hot_ins)

    def forward(self, x):
        if isinstance(x, int):
            x = self.one_hot(x)
        return self.hidden_net(x)
