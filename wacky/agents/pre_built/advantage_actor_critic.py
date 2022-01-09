import torch as th
from torch.nn import functional as F

from wacky.agents import MonteCarloLearner
from wacky.losses import AdvantageLoss, ValueLossWrapper
from wacky.scores import MonteCarloReturns, CalcAdvantages
from wacky.memory import MemoryDict


class AdvantageActorCritic(MonteCarloLearner):

    def __init__(
            self,
            network,
            optimizer: str = 'Adam',
            lr: float = 1.e-3,
            returns_gamma: float = 0.99,
            returns_standardize: bool = False,
            returns_standardize_eps: float = 1.e-07,
            actor_loss_scale_factor: float = 1.0,
            critic_loss_scale_factor: float = 0.5,
            *args, **kwargs
    ):
        super(AdvantageActorCritic, self).__init__(network, optimizer, lr, *args, **kwargs)

        self.network = network

        self.memory = MemoryDict()
        self.remember_rewards = True
        self.reset_memory = True

        self.calc_returns = MonteCarloReturns(returns_gamma, returns_standardize_eps, returns_standardize)
        self.calc_advantages = CalcAdvantages()

        self.actor_loss_fn = AdvantageLoss(actor_loss_scale_factor)
        self.critic_loss_fn = ValueLossWrapper(F.smooth_l1_loss, critic_loss_scale_factor)

    def call(self, state, deterministic=False, remember=True):
        (action, log_prob), value = self.network(state)
        if remember:
            self.memory['log_prob'].append(th.squeeze(log_prob))
            self.memory['values'].append(th.squeeze(log_prob))
        return action

    def learn(self):
        self.memory.stack()
        self.memory['returns'] = self.calc_returns(self.memory)
        self.memory['advantage'] = self.calc_advantages(self.memory).detach()
        loss_actor = self.actor_loss_fn(self.memory)
        loss_critic = self.critic_loss_fn(self.memory)

        loss = loss_actor + loss_critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self):
        self.memory.clear()

    def reward_signal(self, reward):
        self.memory['rewards'].append(reward)


def main():
    import gym
    from wacky import functional as funky
    #env = gym.make('CartPole-v0')
    env = gym.make('CartPole-v0')
    network = funky.actor_critic_net_arch(env.observation_space, env.action_space)
    agent = AdvantageActorCritic(network)
    agent.train(env, 300)
    agent.test(env, 100)


if __name__ == '__main__':
    main()
