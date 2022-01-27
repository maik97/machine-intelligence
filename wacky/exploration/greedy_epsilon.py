import numpy as np
from wacky import functional as funky

class EpsilonGreedy(funky.WackyBase):

    def __init__(self, action_space, eps_init: float = 1.0, eps_discount: float = 0.9995, eps_min: float = 0.1):
        super(EpsilonGreedy, self).__init__()

        self.action_space = action_space
        self.eps = eps_init
        self.eps_init = eps_init
        self.eps_discount = eps_discount
        self.eps_min = eps_min

    def reset(self):
        self.eps = self.eps_init

    def call(self, network, state, deterministic=False):
        if not deterministic:
            self.eps = np.maximum(self.eps * self.eps_discount, self.eps_min)
            deterministic = np.random.random() > self.eps

        if deterministic:
            return np.argmax(network(state).detach().numpy())
        else:
            return self.action_space.sample()

