import numpy as np
from gym import spaces
from wacky import functional as funky
from wacky import networks


def make_distribution_network(in_features, space: spaces.Space):

    # space.Dict, space.Tuple
    if isinstance(space, (spaces.Tuple, spaces.Dict)):
        return [make_distribution_network(in_features, subspace) for subspace in space]

    allowed_spaces = [
        spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.Tuple, spaces.Dict,# spaces.MultiBinary
    ]
    action_shape = funky.decode_gym_space(space, allowed_spaces=allowed_spaces)

    # spaces.MultiBinary - not implemented!
    if isinstance(space, spaces.MultiBinary):
        if isinstance(action_shape, tuple):
            pass
        elif isinstance(action_shape, int):
            pass
        else:
            pass
    # space.Box
    elif isinstance(action_shape, tuple):
        return networks.ContinuousDistributionModule(in_features, action_shape)
    # space.Discrete
    elif isinstance(action_shape, int):
        return networks.DiscreteDistributionModule(in_features, action_shape)
    # space.MultiDiscrete
    elif isinstance(action_shape, np.ndarray):
        return [networks.DiscreteDistributionModule(in_features, int(n)) for n in action_shape]
