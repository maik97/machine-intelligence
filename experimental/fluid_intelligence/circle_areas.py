import numpy as np
from scipy.special import softmax


def calc_circle_area(radius):
    return np.pi * np.square(radius)


weights = np.random.random((10))
weights = softmax(weights)
print(np.sum(calc_circle_area(weights)))
print(calc_circle_area(1))