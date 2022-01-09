import torch as th

from wacky.agents import MonteCarloLearner
from wacky.losses import NoBaselineLoss, WithBaselineLoss
from wacky.scores import MonteCarloReturns
from wacky.memory import MemoryDict


class REINFORCE(MonteCarloLearner):

    def __init__(
            self,
            network,
            optimizer: str = 'Adam',
            lr: float = 0.001,
            returns_gamma: float = 0.99,
            returns_standardize: bool = False,
            returns_standardize_eps: float = 1.e-07,
            loss_scale_factor: float = 1.0,
            baseline: str = None,
            *args, **kwargs
    ):
        super(REINFORCE, self).__init__(network, optimizer, lr, *args, **kwargs)

        self.network = network

        self.memory = MemoryDict()
        self.remember_rewards = True
        self.reset_memory = True

        self.calc_returns = MonteCarloReturns(returns_gamma, returns_standardize_eps, returns_standardize)

        if baseline is None:
            self.loss_fn = NoBaselineLoss(loss_scale_factor)
        else:
            self.loss_fn = WithBaselineLoss(loss_scale_factor, baseline)

    def call(self, state, deterministic=False, remember=True):
        action, log_prob = self.network(state)
        if remember:
            self.memory['log_prob'].append(th.squeeze(log_prob))
        return action

    def learn(self):
        self.memory.stack()
        self.memory['returns'] = self.calc_returns(self.memory)
        loss= self.loss_fn(self.memory)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self):
        self.memory.clear()

    def reward_signal(self, reward):
        self.memory['rewards'].append(reward)

def main():
    import gym
    from gym import spaces
    from wacky import functional as funky
    env = gym.make('CartPole-v0')
    in_features = int(funky.decode_gym_space(env.observation_space, allowed_spaces=[spaces.Box])[0])
    network = funky.make_actor_net(env.action_space, in_features)
    agent = REINFORCE(network)
    agent.train(env, 10000)


if __name__ == '__main__':
    main()
