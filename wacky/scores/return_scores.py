import torch as th
import numpy as np
from wacky import functional as funky
from wacky.memory.running_mean_std import RunningMeanStd

class BaseReturnCalculator(funky.MemoryBasedFunctional):

    def __init__(
            self,
            reward_calc_rms=False,
            reward_eps_rms=1e-4,
            reward_substact_mean=False,
            reward_min=None,
            reward_max=None,
            return_calc_rms=False,
            return_eps_rms=1e-4,
            return_substact_mean=False,
            return_min=None,
            return_max=None,
    ):
        super(BaseReturnCalculator, self).__init__()

        self.reward_calc_rms = reward_calc_rms
        if self.reward_calc_rms:
            self.rms_reward = RunningMeanStd(reward_eps_rms, shape=())
            self.reward_substact_mean = reward_substact_mean
            self.reward_min = reward_min
            self.reward_max = reward_max

        self.return_calc_rms = return_calc_rms
        if self.reward_calc_rms:
            self.rms_return = RunningMeanStd(return_eps_rms, shape=())
            self.return_substact_mean = return_substact_mean
            self.return_min = return_min
            self.return_max = return_max

    def call(self, memory):
        pass

    def rms_normalize_rewards(self, rewards):
        if self.reward_calc_rms:
            return self.rms_reward.normalize(
                rewards, True, self.reward_substact_mean, self.reward_min, self.reward_max
            )
        else:
            return rewards

    def rms_normalize_returns(self, returns):
        if self.return_calc_rms:
            return self.rms_return.normalize(
                returns, True, self.return_substact_mean, self.return_min, self.return_max
            )
        else:
            return returns


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
