import torch as th
import wacky.functional as funky
from wacky.functional.get_optimizer import get_optim


class ReinforcementLearnerArchitecture(funky.WackyBase):

    def __init__(self, network, optimizer: str, lr: float, *args, **kwargs):
        super(ReinforcementLearnerArchitecture, self).__init__()

        self.network = network
        self.optimizer = get_optim(optimizer, self.network.parameters(), lr, *args, **kwargs)

    def call(self, state, deterministic=False):
        pass

    def reset(self):
        if hasattr(self, 'memory') and hasattr(self, 'reset_memory'):
            if self.reset_memory:
                self.memory.clear()

    def reward_signal(self, reward):
        if hasattr(self, 'memory') and hasattr(self, 'remember_reward'):
            if self.remember_reward:
                self.memory['reward'].append(reward)

    def done_signal(self, done):
        if hasattr(self, 'memory') and hasattr(self, 'remember_done'):
            if self.remember_done:
                self.memory['done'].append(done)

    def learn(self):
        pass

    def warm_up(self, env):
        pass

    def train(self, env):
        pass

    def test(self, env):
        pass


class MonteCarloLearner(ReinforcementLearnerArchitecture):

    def __init__(self, network, optimizer: str, lr: float, *args, **kwargs):
        super(MonteCarloLearner, self).__init__(network, optimizer, lr, *args, **kwargs)

    def train(self, env, num_episodes, render=False):

        for e in range(num_episodes):

            self.reset()
            done = False
            state = env.reset()

            while not done:

                state = th.FloatTensor(state).unsqueeze(0)
                action = self.call(state, deterministic=False)
                state, reward, done, _ = env.step(action.item())
                self.reward_signal(reward)
                self.done_signal(done)

                if render:
                    env.render()

            self.learn()

    def test(self, env, num_episodes, render=True):

        for e in range(num_episodes):

            self.reset()
            done = False
            state = env.reset()

            while not done:
                state = th.FloatTensor(state).unsqueeze(0)
                action = self.call(state, deterministic=True)
                state, reward, done, _ = env.step(action.item())

                if render:
                    env.render()
