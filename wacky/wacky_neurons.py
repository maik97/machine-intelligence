import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

import numpy as np
import wacky.functional as funky


class WackyNeuron(nn.Module):
    def __init__(self, n_inputs, n_outputs, optimizer=None):
        super().__init__()

        if optimizer is None:
            optimizer = optim.Adam(self.parameters())

        self.linear = nn.Linear(n_inputs, n_outputs)
        self.state_value = nn.Linear(n_inputs + n_outputs, 1)
        self.confidence_mu = nn.Linear(n_inputs + n_outputs + 1, 1)
        self.confidence_sigma = nn.Linear(n_inputs + n_outputs + 1, 1)

        self.optimizer = optimizer

    def reset(self):

        self.v_list = []
        self.mu_list = []
        self.sigma_list = []

    def forward(self, x):
        outs = F.sigmoid(self.linear(x))

        x = torch.flatten(torch.cat([x, outs], dim=0))
        v = self.state_value(x)

        x = torch.flatten(torch.cat([x, v], dim=0))
        mu = F.tanh(self.confidence_mu(x))
        sigma = F.sigmoid(self.confidence_mu(x))

        self.v_list.append(v)
        self.mu_list.append(mu)
        self.sigma_list.append(sigma)

        return outs

    def learn(self, r):

        v = torch.stack(self.v_list)
        mu = torch.stack(self.mu_list)
        sigma = torch.stack(self.sigma_list)

        returns = funky.n_step_returns(rewards=r, gamma=0.9, eps=np.finfo(np.float32).eps.item())
        advantages = funky.calc_advantages(returns=r, values=v)

        log_probs = funky.calc_log_probs(mu=mu, sigma=sigma)

        policy_losses = funky.adv_actor_critic_loss(log_prob=log_probs, advantage=advantages)
        value_losses = funky.val_l1_smooth_loss(values=v, returns=returns)

        self.optimizer.zero_grad()
        loss = policy_losses.sum() + value_losses.sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        self.reset()


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