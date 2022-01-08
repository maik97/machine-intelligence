import torch as th


def monte_carlo_returns(rewards, gamma=0.99, eps=1e-07, standardize=False):

    future_return = 0
    returns = []
    for r in reversed(rewards):
        future_return = r + gamma * future_return
        returns.insert(0, future_return)

    returns = th.tensor(returns)

    if standardize:
        returns = (returns - returns.mean()) / (returns.std() + eps)

    return returns.detach()


def calc_advantages(returns, values):
    return th.sub(returns, values)
