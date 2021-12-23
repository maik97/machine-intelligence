import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

import numpy as np


class WackyNeuron(nn.Module):
    def __init__(self, n_inputs, n_outputs, optimizer=None):
        super().__init__()

        if optimizer is None:
            optimizer = optim.Adam(self.parameters())

        self.linear = nn.Linear(n_inputs, n_outputs)
        self.confidence = nn.Linear(n_inputs, 1)
        self.optimizer = optimizer



    def forward(self, x):
        return self.linear(x)

    def learn(self):
        policy_losses = self.policy_loss(self.memory)
        value_losses = self.value_loss(self.memory)

        self.optimizer.zero_grad()
        loss = policy_losses.sum() + value_losses.sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()


class HiddenWackyNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1000, 10)

    def forward(self, x):
        return self.linear(x)

class OutputWackyNeuron(nn.Module):
    def __init__(self, n_outputs=10):
        super().__init__()
        self.linear = nn.Linear(1000, n_outputs)

    def forward(self, x):
        return self.linear(x)


class WackyNeuronCluster(nn.Module):

    def __init__(
            self,
            inputs,
            outputs,
            n_hidden,
    ):
        super().__init__()

        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]

        self.inputs = inputs
        self.outputs = outputs

        self.n_hidden = n_hidden

    def


torch.manual_seed(0)  #  for repeatable results
basic_model =nn.Linear()
inp = np.array([[[[1,2,3,4],  # batch(=1) x channels(=1) x height x width
                  [1,2,3,4],
                  [1,2,3,4]]]])
x = torch.tensor(inp, dtype=torch.float)
print('Forward computation thru model:', basic_model(x))