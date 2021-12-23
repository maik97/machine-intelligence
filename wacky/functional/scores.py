import torch as th
import numpy as np
import wacky.functional as funky


class NStepReturns(funky.WackyBase):

    def __init__(self, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.eps = np.finfo(np.float32).eps.item()

    def call(self, memory):
        R = 0
        returns = []
        for r in memory('rewards')[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = th.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns


class Advantage(funky.WackyBase):

    def __init__(self):
        super().__init__()

    def call(self, memory):
        advantages = []
        for i in range(len(memory)):
            advantages.append(memory('returns',i) - memory('values',i))
        return advantages
