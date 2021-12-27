from wacky.functional.base import WackyBase
from wacky.functional.losses import (
	clipped_surrogate_loss, ClippedSurrogateLoss, adv_actor_critic_loss, AdvantageActorCritic,
	val_l1_smooth_loss, ValueL1SmoothLoss)
from wacky.functional.losses import n_step_returns, NStepReturns, calc_advantages, CalcAdvantages