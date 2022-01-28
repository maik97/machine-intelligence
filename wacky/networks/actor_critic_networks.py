import torch.nn as nn
from wacky import backend

class ActorCriticNetwork(nn.Module):

    def __init__(self, actor_net_module, critic_net_module, shared_net_module=None):
        super(ActorCriticNetwork, self).__init__()

        backend.check_type(actor_net_module, nn.Module, 'actor_net_module')
        backend.check_type(critic_net_module, nn.Module, 'critic_net_module')
        backend.check_type(shared_net_module, nn.Module, 'shared_net_module', allow_none=True)

        self.shared_net_module = shared_net_module
        self.actor_net_module = actor_net_module
        self.critic_net_module = critic_net_module

    def forward(self, x):
        if self.shared_net_module is not None:
            x = self.shared_net_module(x)
        return self.actor_net_module(x), self.critic_net_module(x)

    def actor(self, x):
        if self.shared_net_module is not None:
            x = self.shared_net_module(x)
        return self.actor_net_module(x)

    def critic(self, x):
        if self.shared_net_module is not None:
            x = self.shared_net_module(x)
        return self.critic_net_module(x)

    def eval_action(self, x, action):
        if self.shared_net_module is not None:
            x = self.shared_net_module(x)
        val = self.critic_net_module(x)

        if len(self.actor_net_module.layers) > 1:
            for layer in self.actor_net_module.layers[:-2]:
                x = layer(x)
        log_prob = self.actor_net_module.layers[-1].eval_action(x, action)

        return log_prob, val


class DuellingQNetwork(nn.Module):

    def __init__(self, value_net_module, adv_net_module, shared_net_module=None):
        super(DuellingQNetwork, self).__init__()

        self.shared_net_module = shared_net_module
        self.value_net_module = value_net_module
        self.adv_net_module = adv_net_module

    def forward(self, x):
        if self.shared_net_module is not None:
            x = self.shared_net_module(x)
        value = self.value_net_module(x)
        adv = self.adv_net_module(x)
        return value + adv - adv.mean(dim=-1, keepdim=True)
