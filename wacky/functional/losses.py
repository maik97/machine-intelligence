import torch as th


def clipped_surrogate_loss(advantage, old_log_prob, log_prob, clip_range):

    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # ratio between old and new policy, should be one at the first iteration
    ratio = th.exp(log_prob - old_log_prob)

    # clipped surrogate loss
    policy_loss_1 = advantage * ratio
    policy_loss_2 = advantage * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

    return policy_loss, ratio


def adv_actor_critic_loss(log_prob, advantage):
    losses = []
    for i in range(len(log_prob)):
        losses.append(-log_prob[i] * advantage[i])
    return th.stack(losses)
