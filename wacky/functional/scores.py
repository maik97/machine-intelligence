import torch as th
import numpy as np
from wacky import functional as funky


def n_step_returns(rewards, gamma, eps):
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = th.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns

class NStepReturns(funky.WackyBase):

    def __init__(self, gamma=0.99, eps=None):
        super().__init__()
        
        if eps is None:
            eps = np.finfo(np.float32).eps.item()

        self.gamma = gamma
        self.eps = eps

    def call(self, memory):
        return n_step_returns(
            rewards = memory('rewards'),
            gamma = self.gamma,
            eps = self.eps
        )


def calc_advantages(returns, values):
    advantages = []
    for i in range(len(returns)):
        advantages.append(returns[i] - values[i])
    return th.stack(advantages)

class CalcAdvantages(funky.WackyBase):

    def __init__(self):
        super().__init__()

    def call(self, memory):
        return calc_advantages(
            returns = memory('returns'),
            values = memory('values')
        )

