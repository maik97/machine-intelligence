import wacky.functional as funky
from wacky.functional.get_optimizer import get_optim

class ReinforcementLearnerArchitecture(funky.WackyBase):

    def __init__(self, network, optimizer: str, lr:float, *args, **kwargs):
        super(ReinforcementLearnerArchitecture, self).__init__()

        self.network = network
        self.optimizer = get_optim(optimizer, self.network.parameters(), lr, *args, **kwargs)

    def call(self, state, mode=None):
        pass

    def reset(self):
        pass

    def reward_signal(self, r):
        pass

    def learn(self):
        pass