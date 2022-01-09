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
        self.actor_net_module(x)
        self.critic_net_module(x)
        return self.actor_net_module(x), self.critic_net_module(x)


