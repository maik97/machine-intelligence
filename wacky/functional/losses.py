import torch as th
import torch.nn.functional as F

from wacky import functional as funky


def clipped_surrogate_loss(advantage, old_log_prob, log_prob, clip_range):

    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # ratio between old and new policy, should be one at the first iteration
    ratio = th.exp(log_prob - old_log_prob)

    # clipped surrogate loss
    policy_loss_1 = advantage * ratio
    policy_loss_2 = advantage * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

    return policy_loss, ratio

class ClippedSurrogateLoss(funky.WackyBase):

    def __init__(self, clip_range=0.2):
        super().__init__()
        self.clip_range = clip_range

    def call(self, memory):
        return clipped_surrogate_loss(
            advantage = memory('advantage'),
            old_log_prob = memory('old_log_prob'),
            log_prob = memory('log_prob'),
            clip_range = self.clip_range
        )


def adv_actor_critic_loss(log_prob, advantage):
    losses = []
    for i in range(len(log_prob)):
        losses.append(-log_prob[i] * advantage[i])
    return th.stack(losses)

class AdvantageActorCritic(funky.WackyBase):

    def __init__(self):
        super().__init__()

    def call(self, memory):
        return adv_actor_critic_loss(
            log_prob = memory('log_probs'),
            advantage = memory('advantage')
        )

