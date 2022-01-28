import torch as th
import numpy as np

from wacky.agents import ReinforcementLearnerArchitecture
from wacky.losses import ClippedSurrogateLoss, ValueLossWrapper
from wacky.scores import GeneralizedAdvantageEstimation
from wacky.memory import MemoryDict

from wacky import functional as funky


class PPO(ReinforcementLearnerArchitecture):

    def __init__(
            self,
            network,
            optimizer: str = 'Adam',
            lr: float = 0.0003,
            gamma: float = 0.99,
            lamda:float = 0.95,
            actor_loss_scale_factor: float = 1.0,
            critic_loss_scale_factor: float = 0.5,
            epochs: int = 10,
            batch_size: int = 64,
            *args, **kwargs
    ):
        super(PPO, self).__init__(network, optimizer, lr, *args, **kwargs)

        self.network = network

        self.memory = MemoryDict()#.set_maxlen(20_000)
        self.remember_rewards = True
        self.reset_memory = True

        self.returns_and_advantages = GeneralizedAdvantageEstimation(gamma, lamda)

        self.actor_loss_fn = ClippedSurrogateLoss()
        self.critic_loss_fn = ValueLossWrapper(th.nn.SmoothL1Loss(), critic_loss_scale_factor)

        self.epochs = epochs
        self.batch_size = batch_size

    def call(self, state, deterministic=False, remember=True):
        action, log_prob = self.network.actor(state)
        if remember:
            self.memory['old_log_prob'].append(log_prob[0].detach())
            self.memory['states'].append(np.squeeze(state))
            self.memory['actions'].append(action[0].detach())
        return action


    def learn(self):
        loss_a = []
        loss_c = []
        for e in range(self.epochs):
            for batch in self.memory.batch(self.batch_size, shuffle=True):

                log_prob, value = self.network.eval_action(batch['states'], batch['actions'])
                next_values = self.network.critic(batch['next_states'])

                batch['log_prob'] = log_prob
                batch['values'] = value.reshape(-1,1)
                batch['next_values'] = next_values.reshape(-1,1)

                ret, adv = self.returns_and_advantages(batch)
                batch['returns'] = ret.reshape(-1,1)
                batch['advantage'] = adv.detach()

                loss_actor = self.actor_loss_fn(batch)
                loss_critic = self.critic_loss_fn(batch)

                loss = loss_actor + loss_critic
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_a.append(loss_actor.detach().numpy())
                loss_c.append(loss_critic.detach().numpy())
        return np.mean(loss_a), np.mean(loss_c)

    def reset(self):
        self.memory.clear()

    def step(self, env, state, deterministic=False):
        state = th.FloatTensor(state).unsqueeze(0)
        action = self.call(state, deterministic=deterministic)
        if isinstance(action, th.Tensor):
            action = action.detach()[0].numpy()
        state, reward, done, _ = env.step(action)
        #reward -= int(done)
        return state, reward, done

    def train(self, env, num_steps=None, train_interval=2048, render=False):

        done = True
        train_interval_counter = funky.ThresholdCounter(train_interval)
        episode_rewards = funky.ValueTracer()
        for t in range(num_steps):

            if done:
                state = env.reset()
                episode_rewards.sum()

            state, reward, done = self.step(env, state, deterministic=False)

            self.memory['rewards'].append(reward)
            self.memory['dones'].append(done)
            self.memory['next_states'].append(np.squeeze(state))
            episode_rewards(reward)

            if render:
                env.render()

            if train_interval_counter():
                loss_a, loss_c, = self.learn()
                print('steps:', t + 1,
                      'rewards:', episode_rewards.reduce_mean(decimals=3),
                      'prob:', np.round(np.exp(self.memory.numpy('old_log_prob', reduce='mean')), 3),
                      'actor_loss', np.round(loss_a, 4),
                      'critic_loss', np.round(loss_c, 4),
                      )
                self.reset()


def main():
    import gym
    from wacky import functional as funky
    #env = gym.make('CartPole-v0')
    env = gym.make('LunarLanderContinuous-v2')
    network = funky.actor_critic_net_arch(env.observation_space, env.action_space)
    agent = PPO(network)
    agent.train(env, 100_000)
    agent.test(env, 100)


if __name__ == '__main__':
    main()
