from wacky import functional as funky
from wacky import memory as mem

class NoBaselineLoss(funky.MemoryBasedFunctional):
    def __init__(self, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor

    def call(self, memory: [dict, mem.MemoryDict]):
        return self.scale_factor * funky.basic_score_loss(
            score=memory['returns'],
            log_prob=memory['log_prob'],
        )

class WithBaselineLoss(funky.MemoryBasedFunctional):
    def __init__(self, scale_factor=1.0, baseline=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.baseline_calc = baseline # TODO: Implement baseline functions

    def call(self, memory: [dict, mem.MemoryDict]):
        return self.scale_factor * funky.basic_score_loss(
            score=memory['baseline_returns'],
            log_prob=memory['log_prob'],
        )

class AdvantageLoss(funky.MemoryBasedFunctional):

    def __init__(self, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor

    def call(self, memory: [dict, mem.MemoryDict]):
        return self.scale_factor * funky.basic_score_loss(
            score=memory['advantage'],
            log_prob=memory['log_prob'],
        )

class ClippedSurrogateLoss(funky.MemoryBasedFunctional):

    def __init__(self, clip_range: float = 0.2):
        """
        Wrapper for wacky.functional.clipped_surrogate_loss() that uses a Dict or MemoryDict to look up arguments.
        Initializes all necessary function hyperparameters.

        :param clip_range: Hyperparameter for the clipped surrogate loss
        """
        super().__init__()
        self.clip_range = clip_range

    def call(self, memory: [dict, mem.MemoryDict]):
        """
        Calls wacky.functional.clipped_surrogate_loss()

        :param memory: Must have following keys: ['advantage', 'old_log_prob', 'log_prob']
        :return: Returns of wacky.functional.clipped_surrogate_loss()
        """
        return funky.clipped_surrogate_loss(
            advantage=memory['advantage'],
            old_log_prob=memory['old_log_prob'],
            log_prob=memory['log_prob'],
            clip_range=self.clip_range
        )



