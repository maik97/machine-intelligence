import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import LogNormal, Normal
import time

import numpy as np
def n_step_returns(rewards, gamma, eps):
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns

def calc_advantages(returns, values):
    print(returns.shape)
    print(values.shape)
    advantages = []
    for i in range(len(returns)):
        advantages.append(returns[i] - values[i])
    return torch.stack(advantages)

def adv_actor_critic_loss(log_prob, advantage):
    losses = []
    try:
        for i in range(len(log_prob)):
            losses.append(-log_prob[i] * advantage[i])
    except Exception as e:
        print(e)
        print(log_prob)
        print(advantage)
        print(log_prob.shape)
        print(advantage.shape)
        exit()
    return torch.mean(torch.stack(losses), dtype=torch.float64)

def val_l1_smooth_loss(values, returns):
    losses = []
    for i in range(len(values)):
        losses.append(F.smooth_l1_loss(values[i], returns[i]))
    return torch.mean(torch.stack(losses))

class WackyNeuron(nn.Module):
    def __init__(self, n_inputs, n_outputs, optimizer=None, activation_out='sigmoid'):
        super().__init__()

        self.linear = nn.Linear(n_inputs, n_outputs)
        self.state_value = nn.Linear(n_inputs + n_outputs, 1)
        self.confidence = nn.Linear(n_inputs + n_outputs + 1, 1)

        if activation_out=='sigmoid':
            self.activation_out=F.sigmoid
        elif activation_out=='tanh':
            self.activation_out = F.tanh

        if optimizer is None:
            optimizer = optim.Adam(self.parameters())
        self.optimizer = optimizer

        self.reset()

    def reset(self):

        self.r_list = []
        self.v_list = []
        self.prob_list = []

    def reward(self, r):
        self.r_list.append(r)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        outs = self.activation_out(self.linear(x))

        x = torch.flatten(torch.cat([x, outs], dim=0))
        v = self.state_value(x)

        x = torch.flatten(torch.cat([x, v], dim=0))
        prob = F.sigmoid(self.confidence(x))

        new_outs = []
        for i in range(len(outs)):
            if np.random.random() < 0.5:
                new_outs.append(outs[i] + (1 - prob + np.finfo(np.float32).eps.item()))
            else:
                new_outs.append(outs[i] - (1 - prob + np.finfo(np.float32).eps.item()))

        self.v_list.append(v.float())
        self.prob_list.append(prob.float())

        return torch.squeeze(torch.cat(new_outs)).detach()

    def learn(self):
        r = np.array(self.r_list)
        v = torch.squeeze(torch.stack(self.v_list)).float()
        prob = torch.squeeze(torch.stack(self.prob_list)).float()
        log_probs = torch.squeeze(torch.log(prob)).float()

        returns = n_step_returns(rewards=r, gamma=0.9, eps=np.finfo(np.float32).eps.item()).float()
        advantages = calc_advantages(returns=r, values=v).float()

        policy_losses = adv_actor_critic_loss(log_prob=log_probs, advantage=advantages).float()
        value_losses = torch.mean(F.smooth_l1_loss(v, returns)).float()

        print('policy_losses:',policy_losses)
        print('value_losses',value_losses)


        self.optimizer.zero_grad()

        loss = (policy_losses.sum() + value_losses.sum()).float()

        # perform backprop
        if not torch.isnan(loss):
            loss.backward(retain_graph=True)
            self.optimizer.step()


        self.reset()

import abc
class WackyBase(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError()

class WackyNeuronCluster(WackyBase):

    def __init__(self, num_inputs, num_hidden, num_outputs, num_modules):
        super(WackyNeuronCluster, self).__init__()

        self.input_module = WackyNeuron(num_inputs, num_hidden)
        self.hidden_modules = [WackyNeuron((num_modules+1)*num_hidden, num_hidden, activation_out='tanh') for i in range(num_hidden)]
        self.output_module = WackyNeuron((num_modules+1)*num_hidden, num_outputs, activation_out='tanh')
        self.threshold_module = WackyNeuron((num_modules+1)*num_hidden, 1)



        self.num_modules = num_modules
        self.num_hidden = num_hidden

        #torch.autograd.set_detect_anomaly(True)

    def reset(self):
        self.hidden_x = torch.zeros((self.num_modules * self.num_hidden))
        self.all_x = torch.zeros(((self.num_modules + 1) *self.num_hidden))
        self.threshold = torch.tensor(0.5)

    def call(self, x):
        x = self.input_module(x)

        counter = 0
        while counter < 1:
            self.all_x = torch.cat([x, self.hidden_x])
            hiddens = []
            for elem in self.hidden_modules:
                hiddens.append(elem(self.all_x))
            self.hidden_x = torch.cat(hiddens)

            threshold_val = self.threshold_module(self.all_x)
            if threshold_val < self.threshold and counter > 2:
                break
            counter += 1
        for i in range(counter-1):
            self.reward(torch.tensor(0), only_hidden=True)
        return self.output_module(self.all_x)

    def reward(self,r, only_hidden=False):
        if not only_hidden:
            self.input_module.reward(r)
            self.output_module.reward(r)
        self.threshold_module.reward(r)
        for elem in self.hidden_modules:
            elem.reward(r)

    def learn(self):
        self.input_module.learn()
        self.output_module.learn()
        self.threshold_module.learn()
        for elem in self.hidden_modules:
            elem.learn()

import gym

env = gym.make("LunarLanderContinuous-v2")
agent = WackyNeuronCluster(8,10,2,10)
for e in range(10_000):

    state = env.reset()
    agent.reset()
    done = False
    r_sum = []
    while not done:
        actions = agent(state)
        state, r, done, _ = env.step(np.squeeze(actions.detach().numpy() ))
        agent.reward(r)
        r_sum.append(r)
        env.render()
    print()
    agent.learn()
    print(np.sum(r_sum))
    print(e)





