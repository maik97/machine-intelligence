import math
import torch as th
from torch import nn
from torch.nn import functional as F

from wacky.modules import WackyLayer, AdderLayer, WackyModule, WackyGatedRecurrentUnit, WackyNeuronConnections
from wacky import functional as funky


class PulseLayer(WackyModule):

    def __init__(self, in_features, out_features):
        super(PulseLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.layer = WackyLayer(in_features, out_features, module=nn.Linear)
        self.pulse_factor = nn.Parameter(th.tensor(0.9))

        self.reset()

    def reset(self):
        self.pulse = th.tensor(1.0)

    def forward(self, x):
        pulse = th.clip(self.pulse_factor, 0.1, 0.99) * self.pulse
        x = self.layer(x) * pulse
        self.pulse = pulse.clone().detach()
        return x


class PulseNetwork(WackyModule):

    def __init__(self, in_features, out_features, cells=8, passes=10):
        super(PulseNetwork, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.cells = cells
        self.passes = passes

        #self.encoder_cells = WackyGatedRecurrentUnit(out_features, out_features)
        self.encoder_passes = WackyGatedRecurrentUnit(out_features * cells, out_features)
        self.pulse_layers = nn.ModuleList()

        for _ in range(cells):
            self.pulse_layers.append(PulseLayer(in_features + out_features, out_features))

        self.reset_cells()
        self.reset()

    def reset(self):
        self.h_passes = th.zeros((1, self.out_features))

    def reset_cells(self):
        self.h_cells = th.zeros((1, self.out_features))
        for pulse_layer in self.pulse_layers:
            pulse_layer.reset()

    def cells_forward(self, x):

        x_i = self.h_cells
        outs = []
        for pulse_layer in self.pulse_layers:
            x_i = pulse_layer(th.cat([x, x_i], -1))
            outs.append(x_i)
        self.h_cells = x_i
        return th.stack(outs).reshape((1,-1))

    def forward(self, x):
        self.reset_cells()

        for _ in range(self.passes - 1):
            x_i = self.cells_forward(x)
            self.h_passes = self.encoder_passes(x_i, self.h_passes)

        return self.h_passes


def main():
    import gym
    from wacky.networks import MultiLayerPerceptron, ActorNetwork
    from wacky.optimizer import TorchOptimizer
    from wacky.agents import REINFORCE
    from wacky.scores import MonteCarloReturns

    env = gym.make('CartPole-v0')

    in_features = funky.decode_gym_space(env.observation_space)[0]
    network = MultiLayerPerceptron(in_features=in_features)
    #network.append_layer(16, nn.ReLU(), WackyNeuronConnections)
    network.append_layer(64, nn.ReLU(), WackyNeuronConnections)
    network.append_layer(64, nn.ReLU())

    network = ActorNetwork(
        action_space=env.action_space,
        network=network,
    )

    print('\n', network)
    optimizer = TorchOptimizer(
        optimizer='Adam',
        network_parameter=network,
        lr=0.001,
    )

    agent = REINFORCE(network, optimizer, returns_calc=MonteCarloReturns())
    agent.train(env, 1_500)
    agent.test(env, 100)

if __name__ == '__main__':
    main()






