from wacky import functional as funky
from wacky import memory as mem


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


class AdvantageActorCriticLoss(funky.MemoryBasedFunctional):

    def __init__(self):
        super().__init__()

    def call(self, memory: [dict, mem.MemoryDict]):
        return funky.adv_actor_critic_loss(
            log_prob=memory['log_prob'],
            advantage=memory['advantage']
        )
