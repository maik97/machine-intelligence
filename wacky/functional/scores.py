import torch as th
from wacky import functional as funky


def monte_carlo_returns(rewards, gamma=0.99, eps=1e-07, standardize=False):

    future_return = 0
    returns = []
    for r in reversed(rewards):
        future_return = r + gamma * future_return
        returns.insert(0, future_return)

    returns = th.tensor(returns)

    if standardize:
        returns = funky.standardize_tensor(returns, eps)

    return th.unsqueeze(returns, dim=-1)


def calc_advantages(returns, values, eps=1e-07, standardize=False):
    advantages = th.sub(returns, values)
    if standardize:
        advantages = funky.standardize_tensor(advantages, eps)
    return advantages
