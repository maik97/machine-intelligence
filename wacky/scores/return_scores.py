from wacky import functional as funky


class NStepReturns(funky.MemoryBasedFunctional):

    def __init__(self, gamma=0.99, eps=1e-07):
        super().__init__()

        self.gamma = gamma
        self.eps = eps

    def call(self, memory):
        return funky.n_step_returns(
            rewards=memory['rewards'],
            gamma=self.gamma,
            eps=self.eps
        )


class CalcAdvantages(funky.MemoryBasedFunctional):

    def __init__(self):
        super().__init__()

    def call(self, memory):
        return funky.calc_advantages(
            returns=memory['returns'],
            values=memory['values']
        )
