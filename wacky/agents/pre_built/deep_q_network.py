import numpy as np
import torch as th

from wacky.agents import BootstrappingLearner
from wacky.losses import ValueLossWrapper
from wacky.scores import NStepReturns, TemporalDifferenceReturns
from wacky.memory import MemoryDict
from wacky.exploration import DiscountingEpsilonGreedy
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
            lr: float = 0.0001,
            buffer_size=20_000,
            eps_init: float = 1.0,
            eps_discount: float = 0.999995,
            eps_min: float = 0.01,
            returns_type = 'TD',
            returns_gamma: float = 0.99,
            batch_size=64,
            epochs=1,
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

        self.memory = MemoryDict().set_maxlen(buffer_size)
        self.reset_memory = True

        self.epsilon_greedy = DiscountingEpsilonGreedy(
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
        self.epochs = epochs

    def call(self, state, deterministic=False, remember=True):
        action = self.epsilon_greedy(self.network.behavior, state, deterministic)
        if remember:
            self.memory['states'].append(np.squeeze(state))
            self.memory['actions'].append(action)
        return action

    def reset(self):
        pass
        #self.memory.clear()

    def q_for_state_action_pair(self, batch):
        values = th.squeeze(self.network.behavior(batch['states']))
        actions = batch['actions'].numpy()
        values = th.stack([values[i, int(actions[i])] for i in range(len(values))])
        return th.reshape(values, (-1, 1))

    def max_q_next_states(self, batch):
        next_values = self.network.target(batch['next_states']).detach().numpy()
        next_values = th.tensor(np.max(next_values, -1))
        return next_values

    def learn(self):

        for epoch in range(self.epochs):
            for batch in self.memory.batch(self.batch_size):

                batch['values'] = self.q_for_state_action_pair(batch)
                batch['next_values'] = self.max_q_next_states(batch)

                batch['returns'] = self.calc_returns(batch)
                loss = self.loss_fn(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.network.update_target_weights()

    def train(self, env, num_steps, train_interval, render=False):

        done = True
        train_interval_counter = funky.ThresholdCounter(train_interval)
        episode_rewards = funky.ValueTracer()
        for t in range(num_steps):

            if done:
                state = env.reset()
                episode_rewards.sum()

            state = th.FloatTensor(state).unsqueeze(0)
            action = self.call(state, deterministic=False)
            if isinstance(action, th.Tensor):
                action = action.detach()[0].numpy()
            state, reward, done, _ = env.step(action)
            reward -= int(done)

            self.memory['next_states'].append(np.squeeze(state))
            self.memory['rewards'].append(reward)
            self.memory['dones'].append(done)
            episode_rewards(reward)

            if render:
                env.render()

            if train_interval_counter():
                self.learn()
                print('steps:', t,
                      'rewards:', episode_rewards.reduce_mean(decimals=3),
                      'actions:', self.memory.numpy('actions', reduce='mean', decimals=3),
                      'epsilon:', np.round(self.epsilon_greedy.eps, 3),
                )
                self.reset()
                #self.test(env, 1)


def main():
    import gym
    env = gym.make('CartPole-v0')
    # env = gym.make('LunarLanderContinuous-v2')
    agent = DoubleDQN(action_space=env.action_space, observations_space=env.observation_space)
    agent.train(env, 1_000_000, 2064)
    agent.test(env, 100)


if __name__ == '__main__':
    main()
