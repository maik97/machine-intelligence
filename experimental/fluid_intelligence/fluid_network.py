import torch as th
from torch import nn
from wacky import functional as funky
ghj

class FluidLayer(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int) -> None:
        super(FluidLayer, self).__init__()

        self.in_features = in_features
        self.weights_activation = nn.Softmax()
        self.weights = th.nn.Parameter(th.rand((out_features, in_features)))

    def forward(self, x):
        x = self.weights_activation(self.weights) * x
        return th.unsqueeze(th.sum(x, -1), 0)


class ReinforceTestFluidNetwork(nn.Module):

    def __init__(self, observation_space, action_space):
        super(ReinforceTestFluidNetwork, self).__init__()

        self.fluid_layer_1 = FluidLayer(funky.decode_gym_space(observation_space)[0], 64)
        self.fluid_layer_2 = FluidLayer(64, 64)
        self.action_layer = funky.make_distribution_network(64, action_space)

    def forward(self, x):
        x = self.fluid_layer_1(x)
        x = self.fluid_layer_2(x)
        return self.action_layer(x)


def main():
    import gym
    from wacky.agents import REINFORCE

    env = gym.make('CartPole-v0')
    agent = REINFORCE(network=ReinforceTestFluidNetwork(env.observation_space, env.action_space))
    agent.train(env, 1000)
    agent.test(env, 100)


if __name__ == '__main__':
    main()
