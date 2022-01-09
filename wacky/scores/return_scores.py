import torch as th
from wacky import functional as funky


class MonteCarloReturns(funky.MemoryBasedFunctional):

    def __init__(self, gamma=0.99, eps=1e-07, standardize=False):
        super().__init__()

        self.gamma = gamma
        self.eps = eps
        self.standardize = standardize

    def call(self, memory):
        return funky.monte_carlo_returns(
            rewards=memory['rewards'],
            gamma=self.gamma,
            eps=self.eps,
            standardize=self.standardize,
        )



class CalcAdvantages(funky.MemoryBasedFunctional):

    def __init__(self, eps=1e-07, standardize=False):
        super().__init__()

        self.eps = eps
        self.standardize = standardize

    def call(self, memory):
        return funky.calc_advantages(
            returns=memory['returns'],
            values=memory['values'],
            eps=self.eps,
            standardize=self.standardize,
        )
