import abc
import numpy as np
from wacky import functional as funky

class EpsilonGreedyBase(funky.WackyBase):

    def __init__(self, action_space, eps_init: float = 1.0):
        super(EpsilonGreedyBase, self).__init__()
        self.action_space = action_space
        self.eps_init = eps_init
        self.reset()

    def reset(self):
        self.eps = self.eps_init

    @abc.abstractmethod
    def step(self):
        pass

    def call(self, network, state, deterministic=False, epsilon_step=True):
        if not deterministic:
            if epsilon_step:
                self.step()
            deterministic = np.random.random() > self.eps

        if deterministic:
            return np.argmax(network(state).detach().numpy())
        else:
            return self.action_space.sample()


class DiscountingEpsilonGreedy(EpsilonGreedyBase):

    def __init__(self, action_space, eps_init: float = 1.0, eps_discount: float = 0.9995, eps_min: float = 0.1):
        super(DiscountingEpsilonGreedy, self).__init__(action_space, eps_init)

        self.eps_discount = eps_discount
        self.eps_min = eps_min

    def step(self):
        self.eps = np.maximum(self.eps * self.eps_discount, self.eps_min)


class InterpolationEpsilonGreedy(EpsilonGreedyBase):

    def __init__(
            self,
            action_space,
            eps_interpolation: str = 'linear',
            eps_init: float = 1.0,
            eps_min: float = 0.1,
            ramp_point_a: float=0.0,
            ramp_point_b: float = 0.0,
    ):
        super(InterpolationEpsilonGreedy, self).__init__(action_space, eps_init)

        self.interpolation = funky.get_ramp_interpolator(
            val_a=eps_init,
            val_b=eps_min,
            point_a=ramp_point_a,
            point_b=ramp_point_b,
            kind=eps_interpolation,
        )
        self.eps_min = eps_min

    def step(self, point):
        self.eps = self.interpolation(point)

    def call(self, network, state, deterministic=False, epsilon_step=True, point=None):
        if epsilon_step and not deterministic and point is not None:
            self.step(point)
        return super().call(network, state, deterministic=deterministic, epsilon_step=False)


