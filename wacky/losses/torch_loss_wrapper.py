from wacky import functional as funky
from wacky import memory as mem


class ValueLossWrapper(funky.MemoryBasedFunctional):
    def __init__(self, loss_fn, scale_factor=1.0):
        super().__init__()
        self.loss_fn = loss_fn
        self.scale_factor = scale_factor

    def call(self, memory: [dict, mem.MemoryDict], *args, **kwargs):
        return self.scale_factor * self.loss_fn(memory['returns'], memory['values'], *args, **kwargs).sum()
