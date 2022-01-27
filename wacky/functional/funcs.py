import numpy as np


def standardize_tensor(vals, eps=1e-08):
    return (vals - vals.mean()) / (vals.std() + eps)


class ThresholdCounter:

    def __init__(self, threshold, init_val=0, min_val=None):
        self.threshold = threshold
        self.count = init_val
        self.init_val = init_val
        self.min_val = min_val

    def __call__(self, amount=1):
        self.count_up(amount)
        if self.count >= self.threshold:
            self.reset()
            return True
        else:
            return False

    def reset(self):
        self.count = self.init_val

    def count_up(self, amount=1):
        self.count +=  amount
        if self.min_val is not None:
            self.count = self.min_val if self.count < self.min_val else self.count

    def count_down(self, amount=1):
        self.count -=  amount
        if self.min_val is not None:
            self.count = self.min_val if self.count < self.min_val else self.count


class ValueTracer:

    def __init__(self):
        self.clear()

    def __call__(self, val):
        self.append(val)

    def reset(self):
        self.arr = np.array([])

    def clear(self):
        self.reset()
        self.mean_arrs  = np.array([])
        self.sum_arrs  = np.array([])

    def append(self, val):
        self.arr = np.append(self.arr, val)

    def mean(self, reset=True):
        mean_arr = np.mean(self.arr)
        if reset:
            self.reset()
        self.mean_arrs = np.append(self.mean_arrs, mean_arr)
        return mean_arr

    def sum(self, reset=True):
        sum_arr = np.sum(self.arr)
        if reset:
            self.reset()
        self.sum_arrs = np.append(self.sum_arrs, sum_arr)
        return sum_arr

    def mean_of_sums(self, clear=True):
        print()
        mean_sum_arrs = np.mean(self.sum_arrs)
        if clear:
            self.clear()
        return mean_sum_arrs

    def sum_of_means(self, clear=True):
        sum_mean_arrs = np.sum(self.mean_arrs)
        if clear:
            self.clear()
        return sum_mean_arrs

