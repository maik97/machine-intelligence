import math
import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule, VariableOutFeaturesLayer, SharedWeightEncoder, VariableInFeaturesLayer
from wacky import functional as funky


class ChannelModule(WackyModule):

    def __init__(
            self,
            in_features,
            out_features,
            max_in_features=300,
            max_out_features=300,
            n_dims=3,
            shared_net=None,
            activation=None
    ):
        super(ChannelModule, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.n_dims = n_dims

        self.in_encoder = VariableInFeaturesLayer(
            max_in_features=max_in_features,
            out_features=in_features,
            n_dims=n_dims,
            activation=nn.ReLU(),
            shared_net=shared_net,
        )

        self.recurrent_cell = nn.GRUCell(in_features, out_features)

        self.out_decoder = VariableOutFeaturesLayer(
            in_features=out_features,
            max_out_features=max_out_features,
            n_dims=n_dims,
            activation=nn.ReLU(),
            shared_net=shared_net,
        )

        self.learn()

    def learn(self, *args, **kwargs):
        self.h = th.zeros(1, self.out_features)

    def forward(self):