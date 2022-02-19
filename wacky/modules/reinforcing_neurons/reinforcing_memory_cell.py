import math
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from gym import spaces

from wacky.modules import CellAgent, WackyModule
from wacky.agents import REINFORCE, make_REINFORCE
from wacky.optimizer import TorchOptimizer
from wacky.networks import ActorNetwork, MultiLayerPerceptron
from wacky import functional as funky
from wacky.losses import ValueLossWrapper


class RecurrentAgent(WackyModule):

    def __init__(self, in_features, out_features, local_network=None, atom_size=101, v_min=-0.5, v_max=0.5):
        super(RecurrentAgent, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        agent_in_features = in_features + out_features * 2
        if local_network is None:
            local_network = [int(agent_in_features * 1.5), int(atom_size * 1.5)]

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(agent_in_features,))
        self.action_space = spaces.Discrete(atom_size)
        self.agent = CellAgent(make_REINFORCE(self, local_network, standardize_returns=False))
        self.support = th.linspace(v_min, v_max, atom_size)

        self.reset_dummy_parameters()
        self.reset()

    def reset(self):
        self.hidden_state = th.zeros(self.out_features)

    def clone_dummy_parameters(self) -> None:
        self.old_weight = self.dummy_weight.clone().detach()
        self.old_bias = self.dummy_bias.clone().detach()

    def reset_dummy_parameters(self) -> None:
        self.dummy_weight = nn.Parameter(th.empty((self.out_features, self.in_features)))
        self.dummy_bias = nn.Parameter(th.empty(self.out_features))
        nn.init.kaiming_uniform_(self.dummy_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dummy_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.dummy_bias, -bound, bound)
        self.clone_dummy_parameters()

    def dummy_parameter_changes(self, mode='abs', reduce='sum'):
        dummy_w = th.abs(self.dummy_weight.clone().detach() - self.old_weight).sum()
        dummy_b = th.abs(self.dummy_bias.clone().detach() - self.old_bias).sum()
        self.clone_dummy_parameters()
        return dummy_w + dummy_b

    def dummy_gradient_trace(self, x, outs):
        dummy_outputs = th.add(th.mm(x, self.dummy_weight.t()), self.dummy_bias)
        cloned_dummy_outputs = dummy_outputs.clone().detach()
        mul_factors = th.div(outs.clone().detach(), cloned_dummy_outputs).detach()
        return th.mul(dummy_outputs, mul_factors)

    def forward(self, x):

        for i in range(self.out_features):
            agent_inputs = th.cat(
                [th.squeeze(x), self.hidden_state, F.one_hot(th.tensor(i), self.out_features)],
                dim=-1
            ).reshape(1, -1)
            action = self.agent(agent_inputs, deterministic=False)
            change = self.support[action]
            self.hidden_state[i] = self.hidden_state[i] + change

        return self.dummy_gradient_trace(x, self.hidden_state)

    def learn(self, loss: th.Tensor):
        diff = - self.dummy_parameter_changes() * loss.detach()
        self.agent.memory['rewards'] = [th.tensor(1.0)] * self.agent.memory.global_keys_len
        self.agent.memory['rewards'][-1] = th.squeeze(diff.detach()) # th.tensor(-1.0)#
        loss = self.agent.learn()
        self.agent.reset()
        self.reset()
        return loss


class TestApproximationNetwork(funky.WackyBase):

    def __init__(self, network, loss_fn=None, optimizer='Adam', lr=0.00001):
        super(TestApproximationNetwork, self).__init__()
        self.network = network
        self.loss_fn = loss_fn if loss_fn is not None else th.nn.SmoothL1Loss()
        self.optimizer = TorchOptimizer(
            optimizer=optimizer,
            network_parameter=network,
            lr=lr,
        )

    def call(self, x):
        return self.network(x)

    def train_step(self, x, target):
        y = self.call(x)
        loss = self.loss_fn(y, target)
        self.optimizer.apply_loss(loss)
        self.network.learn(loss=loss)
        return loss


def test_math_approximator(num_epochs=10_000):
    network = MultiLayerPerceptron(in_features=2)
    network.append_layer(32, None, RecurrentAgent)
    network.append_layer(32, None, RecurrentAgent)
    network.append_layer(32, nn.ReLU())
    network.append_layer(1, None)

    model = TestApproximationNetwork(network)

    for e in range(num_epochs):
        losses = []
        for i in range(100):
            x = np.random.random(2)
            target = np.sum(x)
            loss = model.train_step(
                th.tensor(x, dtype=th.float64).reshape(1,-1).float(),
                th.tensor(target, dtype=th.float64).reshape(1,-1).float()
            )
            losses.append(loss.detach().numpy())
        print('step', 100*e, 'loss', np.mean(losses))


def main():
    test_math_approximator()
    exit()
    import gym
    env = gym.make('CartPole-v0')

    in_features = funky.decode_gym_space(env.observation_space)[0]
    reinforcing_network = MultiLayerPerceptron(in_features=in_features)
    reinforcing_network.append_layer(16, None, RecurrentAgent)
    reinforcing_network.append_layer(64, nn.ReLU())
    reinforcing_network.append_layer(64, nn.ReLU())
    # reinforcing_network.append_layer(8, None, LayerReinforcingNeurons)

    network = ActorNetwork(
        action_space=env.action_space,
        network=reinforcing_network,
    )

    print('\n', network)
    optimizer = TorchOptimizer(
        optimizer='Adam',
        network_parameter=network,
        lr=0.001,
    )

    agent = REINFORCE(network, optimizer)
    agent.train(env, 10000)
    agent.test(env, 100)

if __name__ == '__main__':
    main()
