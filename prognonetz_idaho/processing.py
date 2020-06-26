import numpy as np


class Processing(object):
    """
    Processing class
    """
    def __init__(self, forward_function, mask=None, reverse_function=None):
        self.forward = forward_function
        self.reverse = reverse_function
        self.mask = mask

    def process(self, x):
        if self.mask is None:
            return self.forward(x)
        else:
            tmp = x[..., self.mask]
            tmp = self.forward(tmp)
            x[..., self.mask] = tmp[..., :]
            return x

    def revert(self, x):
        if self.mask is None:
            return self.reverse(x)
        else:
            tmp = x[..., self.mask]
            tmp = self.reverse(tmp)
            x[..., self.mask] = tmp[..., :]
            return x


class Identity(Processing):
    """
    Identity processing.
    """
    def __init__(self):
        Processing.__init__(self, self._no_processing, None, self._no_processing)

    def _no_processing(self, x):
        return x


class Normalize(Processing):
    """
    Min-max normalization
    """

    def __init__(self, minimum_value, maximum_value):
        # create mask for selective processing
        mask = None
        if type(minimum_value) is list and type(maximum_value) is list:
            indices = [i for i, (x, y) in enumerate(zip(minimum_value, maximum_value)) if x is not None and y is not None]
            minimum_value = [float(minimum_value[i]) for i in indices]
            maximum_value = [float(maximum_value[i]) for i in indices]
            mask = indices

        Processing.__init__(self, self._normalize, mask, self._denormalize)
        self.minimum_value = np.array(minimum_value)
        self.maximum_value = np.array(maximum_value)

    def _normalize(self, x):
        return (x - self.minimum_value)/(self.maximum_value - self.minimum_value)

    def _denormalize(self, x):
        return x * (self.maximum_value - self.minimum_value) + self.minimum_value

class Standardize(Processing):
    """
    Standardization to between -1 and 1.
    """

    def __init__(self, mean=None, standard_deviation=None):
        # create mask for selective processing
        mask = None
        if type(mean) is list and type(standard_deviation) is list:
            indices = [i for i, (x, y) in enumerate(zip(mean, standard_deviation)) if x is not None and y is not None]
            mean = [float(mean[i]) for i in indices]
            standard_deviation = [float(standard_deviation[i]) for i in indices]
            mask = indices

        Processing.__init__(self, self._standardize, mask, self._destandardize)
        self.mean = mean
        self.standard_deviation = standard_deviation

    def _standardize(self, x):
        return (x - self.mean)/self.standard_deviation

    def _destandardize(self, x):
        return x * self.standard_deviation + self.mean


class Vector(object):
    """
    Vector class
    """
    def __init__(self, magnitude, direction):
        self.magnitude = magnitude
        self.direction = direction
