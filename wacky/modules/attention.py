import math
import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyModule, WackyLayer, VariableInFeaturesLayer


class VaryingAttentionHead(WackyModule):

    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.qkv_proj = VariableInFeaturesLayer(in_features, 3 * out_features, *args, **kwargs)
        self.out_proj = WackyLayer(out_features, out_features, activation=nn.Sigmoid())

    def attention(self, q, v, k):
        d_k = q.shape[-1]
        att = q * k
        att = att / math.sqrt(d_k)
        att = F.softmax(att, dim=-1)
        return att * v

    def forward(self, x):
        qkv = self.qkv_proj(x.reshape(1, -1))
        q, k, v = qkv.chunk(3, dim=-1)
        x = self.attention(q, v, k)
        return self.out_proj(x)
