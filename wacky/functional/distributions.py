import numpy as np
from gym import spaces
from wacky.functional.gym_space_decoder import decode_gym_space

def make_continuous_distribution(action_shape: tuple):

def make_continuous_distribution(action_shape: tuple):

def make_action_distribution(space: spaces.Space):

    # space.Dict, space.Tuple
    if isinstance(space, (spaces.Tuple, spaces.Dict)):
        return [make_action_distribution(subspace) for subspace in space]

    allowed_spaces = [
        spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.Tuple, spaces.Dict,# spaces.MultiBinary
    ]
    action_shape = decode_gym_space(space, allowed_spaces=allowed_spaces)

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

    # space.Discrete
    elif isinstance(action_shape, int):

    # space.MultiDiscrete
    elif isinstance(action_shape, np.ndarray):



