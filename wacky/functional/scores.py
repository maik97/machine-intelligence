import torch as th
from wacky import functional as funky


def monte_carlo_returns(rewards, gamma=0.99, eps=1e-07, standardize=False):

    future_return = 0.0
    returns = []
    for r in reversed(rewards):
        future_return = r + gamma * future_return
        returns.insert(0, future_return)

    returns = th.tensor(returns)

    if standardize:
        returns = funky.standardize_tensor(returns, eps)

    return th.unsqueeze(returns, dim=-1)


def generalized_returns(rewards, dones, values, next_values, gamma=0.99, lamda=0.95, eps=1e-07, standardize=False):
    g = 0.0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * dones[i] - values[i]
        g = delta + gamma * lamda * int(dones[i]) * g
        future_return = g + values[i]
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


def generalized_advantage_estimation(
        rewards, dones, values, next_values, gamma=0.99, lamda=0.95, eps=1e-07,
        standardize_returns=False, standardize_advantages=False
):

    returns = generalized_returns(rewards, dones, values, next_values, gamma, lamda, eps, standardize_returns)
    advantages = calc_advantages(returns, values, eps, standardize_advantages)

    return returns, advantages

