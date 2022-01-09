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
        pass

    def reward_signal(self, reward):
        pass

    def done_signal(self, done):
        pass

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
                action = self.call(state, deterministic=False)[0]
                state, reward, done, _ = env.step(action.item())
                self.reward_signal(reward)
                self.done_signal(done)

                if render:
                    env.render()

            self.learn()
            print('rewards:',self.memory['rewards'].sum().numpy(),
                  'probs:', th.exp(self.memory['log_prob'].detach()).mean().numpy()
            )

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
