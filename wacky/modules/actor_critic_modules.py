import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wacky.functional as funky
from wacky.backend.get_optimizer import get_optim

class ReinforcementLearnerArchitecture(funky.WackyBase):

    def __init__(self, network, optimizer: str, lr:float, *args, **kwargs):
        super(ReinforcementLearnerArchitecture, self).__init__()

        self.network = network
        self.optimizer = get_optim(optimizer, self.network.parameters(), lr, *args, **kwargs)

    def call(self, state, mode=None):
        pass

    def reset(self):
        pass

    def reward_signal(self, r):
        pass

    def learn(self):
        pass


class ActorCriticArchitecture(ReinforcementLearnerArchitecture):

    def __init__(
            self,
            network: nn.Module,
            optimizer: str,
            lr: float,
            distribution,
            returns_fn,
            advantages_fn,
            actor_loss_fn,
            critic_loss_fn,
            *args, **kwargs):
        super(ActorCriticArchitecture, self).__init__(network, optimizer, lr, *args, **kwargs)

        self.returns_fn = returns_fn
        self.advantages_fn = advantages_fn
        self.actor_loss_fn = actor_loss_fn
        self.critic_loss_fn = critic_loss_fn

    def call(self, state, mode=None):
        pass

    def reset(self):
        pass

    def reward_signal(self, r):
        pass

    def learn(self):
        pass


class AdvantageActorCriticModule(nn.Module):
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

        returns = funky.n_step_returns(rewards=r, gamma=0.9, eps=np.finfo(np.float32).eps.item()).float()
        advantages = funky.calc_advantages(returns=r, values=v).float()

        policy_losses = funky.adv_actor_critic_loss(log_prob=log_probs, advantage=advantages).float()
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


class ExperimentalAdvantageActorCriticModule(nn.Module):
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

        returns = funky.n_step_returns(rewards=r, gamma=0.9, eps=np.finfo(np.float32).eps.item()).float()
        advantages = funky.calc_advantages(returns=r, values=v).float()

        policy_losses = funky.adv_actor_critic_loss(log_prob=log_probs, advantage=advantages).float()
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