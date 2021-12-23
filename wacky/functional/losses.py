import torch as th
import torch.nn.functional as F

import wacky.functional as funky

class ClippedSurrogateLoss(funky.WackyBase):

    def __init__(self, clip_range=0.2):
        super().__init__()
        self.clip_range = clip_range

    def call(self, rollout_data, log_prob):
        advantages = rollout_data.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - rollout_data.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        #pg_losses.append(policy_loss.item())
        #clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        #clip_fractions.append(clip_fraction)

        return policy_loss


class AdvantageActorCritic(funky.WackyBase):

    def __init__(self):
        super().__init__()

    def call(self, memory):
        losses = []
        for i in range(len(memory)):
            losses.append(-memory('log_probs', i) * memory('advantage', i))
        return th.stack(losses)


class ValueL1SmoothLoss(funky.WackyBase):

    def __init__(self):
        super().__init__()

    def call(self, memory):
        losses = []
        for i in range(len(memory)):
            losses.append(F.smooth_l1_loss(memory('values', i), memory('returns',i)))
        return losses
