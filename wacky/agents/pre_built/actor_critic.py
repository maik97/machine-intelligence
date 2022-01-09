import torch as th
from torch.nn import functional as F

from wacky.agents import MonteCarloLearner
from wacky.losses import NoBaselineLoss, WithBaselineLoss, ValueLossWrapper
from wacky.scores import MonteCarloReturns
from wacky.memory import MemoryDict


class ActorCritic(MonteCarloLearner):

    def __init__(
            self,
            network,
            optimizer: str = 'Adam',
            lr: float = 0.01,
            returns_gamma: float = 0.99,
            returns_standardize: bool = False,
            returns_standardize_eps: float = 1.e-07,
            actor_loss_scale_factor: float = 1.0,
            critic_loss_scale_factor: float = 0.5,
            baseline: str = None,
            *args, **kwargs
    ):
        super(ActorCritic, self).__init__(network, optimizer, lr, *args, **kwargs)

        self.network = network

        self.memory = MemoryDict()
        self.remember_rewards = True
        self.reset_memory = True

        self.calc_returns = MonteCarloReturns(returns_gamma, returns_standardize_eps, returns_standardize)

        if baseline is None:
            self.actor_loss_fn = NoBaselineLoss(actor_loss_scale_factor)
        else:
            self.actor_loss_fn = WithBaselineLoss(actor_loss_scale_factor, baseline)

        self.critic_loss_fn = ValueLossWrapper(F.smooth_l1_loss, critic_loss_scale_factor)

    def call(self, state, deterministic=False, remember=True):
        action, log_prob = self.network(state, deterministic)
        if remember:
            self.memory['log_prob'].append(log_prob)
        return action

    def learn(self):
        self.memory['returns'] = self.calc_returns(self.memory)
        loss_actor = self.actor_loss_fn(self.memory)
        loss_critic = self.critic_loss_fn(self.memory)

        loss = loss_actor + loss_critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
