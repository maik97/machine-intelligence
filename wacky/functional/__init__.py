from wacky.functional.base import WackyBase
from wacky.functional.get_optimizer import get_optim


from wacky.functional.losses import clipped_surrogate_loss, ClippedSurrogateLoss
from wacky.functional.losses import adv_actor_critic_loss, AdvantageActorCritic

from wacky.functional.scores import n_step_returns, NStepReturns
from wacky.functional.scores import calc_advantages, CalcAdvantages
