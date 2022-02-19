import abc

import torch as th
from torch import nn

from wacky.modules import WackyModule
from wacky.agents import ReinforcementLearnerArchitecture
from wacky import functional as funky


class ReinforcingModule(WackyModule):

    def __init__(self):
        super(ReinforcingModule, self).__init__()
        self.gradiant_reward = nn.Parameter(th.empty((1)))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward method implementation. Must be called by subclass in sub method of forward()."""
        return self.gradiant_reward * x

    def reset_gradient_reward(self):
        """Sets parameter gradient_reward to 1.0"""
        self.gradiant_reward = nn.Parameter(th.tensor(0.0))#(self.gradiant_reward / self.gradiant_reward)

    @abc.abstractmethod
    def learn(self,  *args, **kwargs):
        """Train ReinforcingModule with Reinforcement Learning"""


class ReinforcingCell(WackyModule):

    def __init__(self):
        super(ReinforcingCell, self).__init__()

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        """Train CellAgent(s) with Reinforcement Learning"""


class CellAgent(funky.WackyBase):

    def __init__(self, agent: ReinforcementLearnerArchitecture):
        super(CellAgent, self).__init__()
        self.agent = agent

    def __call__(self, x: th.Tensor, *args, **kwargs) -> th.Tensor:
        return self.call(x.detach(), *args, **kwargs).detach()

    def call(self, x, *args, **kwargs):
        return self.agent(x, *args, **kwargs)

    def learn(self, *args, **kwargs):
        return self.agent.learn()

    def reward_signal(self, reward):
        if hasattr(self.agent, 'memory'):
            if not self.agent.memory.stacked:
                self.agent.memory.append(reward)
        elif funky.has_method(self.agent, 'reward_signal'):
            self.agent.reward_signal(reward)
        else:
            raise Exception('Agent has neither a memory attribute nor a reward_signal() method.')

    @property
    def memory(self):
        return self.agent.memory

    def reset(self):
        self.agent.reset()