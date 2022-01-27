import numpy as np
import torch as th

from wacky.agents import BootstrappingLearner
from wacky.losses import ValueLossWrapper
from wacky.scores import NStepReturns, TemporalDifferenceReturns
from wacky.memory import MemoryDict
from wacky.exploration import EpsilonGreedy
from wacky.networks import DoubleNetworkWrapper

from wacky import functional as funky


class DoubleDQN(BootstrappingLearner):

    def __init__(
            self,
            action_space,
            observations_space,
            network=None,
            polyak=0.995,
            optimizer: str = 'Adam',
            lr: float = 0.0007,
            eps_init: float = 1.0,
            eps_discount: float = 0.9995,
            eps_min: float = 0.1,
            returns_type = 'TD',
            returns_gamma: float = 0.99,
            batch_size=64,
            *args, **kwargs
    ):

        self.network = DoubleNetworkWrapper(
            make_net_func=funky.make_q_net,
            polyak=polyak,
            in_features=observations_space,
            out_features=action_space,
            q_net=network
        )

        super(DoubleDQN, self).__init__(self.network, optimizer, lr, *args, **kwargs)

        self.memory = MemoryDict()
        self.reset_memory = True

        self.epsilon_greedy = EpsilonGreedy(
            action_space,
            eps_init=eps_init,
            eps_discount=eps_discount,
            eps_min=eps_min
        )

        if returns_type == 'TD':
            self.calc_returns = TemporalDifferenceReturns(returns_gamma)
        elif returns_type == 'N-Step':
            self.calc_returns = NStepReturns(returns_gamma)

        self.loss_fn = ValueLossWrapper(th.nn.SmoothL1Loss())

        self.batch_size = batch_size

    def call(self, state, deterministic=False, remember=True):
        action = self.epsilon_greedy(self.network.behavior, state, deterministic)
        if remember:
            self.memory['states'].append(np.squeeze(state))
            self.memory['actions'].append(action)
        return action

    def next_state(self, state):
        self.memory['next_states'].append(np.squeeze(state))

    def reward_signal(self, reward):
        self.memory['rewards'].append(reward)

    def done_signal(self, done):
        self.memory['dones'].append(done)

    def action_as_idx(self, batch, i):
        return int(batch['actions'].numpy()[i])


    def max_q_next_states(self, batch):
        next_values = np.max(self.network.target(batch['next_states']).detach().numpy(), -1)
        return th.tensor(next_values)

    def learn(self):

        self.memory.stack()

        for batch in self.memory.batch(self.batch_size):

            values = th.squeeze(self.network.behavior(batch['states']))
            values = th.stack([values[i, self.action_as_idx(batch, i)] for i in range(len(values))])
            batch['values'] = th.reshape(values, (-1, 1))
            batch['next_values'] = self.max_q_next_states(batch)

            batch['returns'] = self.calc_returns(batch)
            loss = self.loss_fn(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.network.update_target_weights()

    def reset(self):
        self.memory.clear()


def main():
    import gym
    env = gym.make('CartPole-v0')
    # env = gym.make('LunarLanderContinuous-v2')
    agent = DoubleDQN(action_space=env.action_space, observations_space=env.observation_space)
    agent.train(env, 1_000_000, 2064)
    agent.test(env, 100)


if __name__ == '__main__':
    main()
