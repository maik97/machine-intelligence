import math
import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, WackyModule, SharedWeightEncoder
from wacky import functional as funky

class ApproximateActivation(WackyModule):

    def __init__(self, n_dims=3, shared_net=None):
        super(ApproximateActivation, self).__init__()

        if shared_net is not None:
            self.hidden_net = shared_net
        else:
            self.hidden_net = SharedWeightEncoder(n_dims=n_dims)

        self.squashing_net = WackyLayer(
            in_features=self.hidden_net.out_features,
            out_features=3,
            module=nn.Linear,
        )

        self.threshold_net = WackyLayer(
            in_features=self.hidden_net.out_features,
            out_features=3,
            module=nn.Linear,
        )

        self.reverse_threshold_net = WackyLayer(
            in_features=self.hidden_net.out_features,
            out_features=3,
            module=nn.Linear,
        )

    @staticmethod
    def decode_binary(binary):
        binary = F.tanh(binary)
        binary = th.sign(binary)
        binary = th.relu(binary)
        return binary, (1 - binary)

    def squashing(self, x, binary, mid, max_diff):
        squash_it, dont_squash_it = self.decode_binary(binary)
        squashed = F.tanh(x) * max_diff + mid
        return squash_it * squashed + dont_squash_it * x

    def threshold(self, x, binary, threshold, value):
        threshold_it, dont_threshold_it = self.decode_binary(binary)
        return threshold_it * F.threshold(x, threshold, value) + dont_threshold_it * x

    def reverse_threshold(self, x, binary, threshold, value):
        threshold_it, dont_threshold_it = self.decode_binary(binary)
        return - threshold_it * F.threshold(x * -1, threshold * -1, value * -1) + dont_threshold_it * x

    def calc_activation(self, neuron_emb):
        a = self.hidden_net(neuron_emb)
        return self.squashing_net(a), self.threshold_net(a), self.reverse_threshold_net(a)

    def forward(self, x: th.Tensor, neuron_emb) -> th.Tensor:
        s, t, r_t = self.calc_activation(neuron_emb)

        s_b, s_m, s_m_d = s.chunk(3, dim=-2)
        x = self.squashing(x, s_b, s_m, s_m_d)

        t_b, t_t, t_v = t.chunk(3, dim=-2)
        x = self.threshold(x, t_b, t_t, t_v)

        r_t_b, r_t_t, r_t_v = r_t.chunk(3, dim=-2)
        x = self.reverse_threshold(x, r_t_b, r_t_t, r_t_v)

        return x
