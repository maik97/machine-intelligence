import copy
import torch as th
import numpy as np

from wacky import functional as funky
from wacky.agents import make_REINFORCE


class OverfitIt:

    def __init__(self, env):
        self.env = env
        self.agent = make_REINFORCE(env)

        # initial parameter because each network should evolve from the same starting point
        self.init_params = self.agent_parameter()

        # make parameter predictive network based on input of episode starting state
        self.network = funky.maybe_make_network(
            network=[128, 512],  # hidden net
            in_features=env.observation_space,  # input size
            activation=th.nn.ReLU()  # hidden activation
        )
        self.network.append_layer(
            units=len(self.init_params),  # output size
            activation=None  # no activation for output layer
        )
        print(self.network)

    def agent_parameter(self):
        params = funky.maybe_get_network_params(self.agent.network)
        param_list = []
        for param in params:
            param_list.append(param.clone().detach().flatten())
        return th.cat(param_list)

    def overfit(self, n_episode=100, render=True):
        # evtl. das network für jeden overfit von dem pred param net vorhersehen lassen
        # und dann am ende loss mit den unterschieden wie es nach dem overfitting aussah

        #start_state = self.env.reset()
        #action = self.agent.call(th.FloatTensor(start_state).unsqueeze(0), deterministic=True).detach()[0]
        #start_state, reward, done, _ = self.env.step(action.numpy())
        #self.agent.network.reset()


        start_state = self.env.reset()
        self.env.render()

        for e in range(n_episode):
            temp_env = copy.deepcopy(self.env)
            done = False
            state = start_state
            self.agent.reset()

            while not done:

                state = th.FloatTensor(state).unsqueeze(0)
                action = self.agent.call(state, deterministic=False).detach()[0]
                state, reward, done, _ = temp_env.step(action.numpy())
                self.agent.memory['rewards'].append(reward)

                if render:
                    temp_env.render()
            temp_env.close()
            self.agent.network.reset()

            loss = self.agent.learn()
            print('episode:', e,
                  'rewards:', self.agent.memory['rewards'].sum().numpy(),
                  'probs:', np.round(th.exp(self.agent.memory['log_prob'].detach()).mean().numpy(), 4),
                  'loss:', np.round(loss.numpy(), 2),
                  )



def main():
    import gym
    env = gym.make('LunarLanderContinuous-v2')
    overfitted = OverfitIt(env)
    overfitted.overfit()
    #agent = make_REINFORCE(env, network=[16,16])
    #agent.train(env, 10000)
    #agent.test(env, 100)


if __name__ == '__main__':
    main()