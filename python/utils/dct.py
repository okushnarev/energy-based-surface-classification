from typing import Iterable

import numpy as np
import scipy as sci


class DCT:
    """
    A class that performs a Discrete Cosine Transform (DCT) on discrete data
    and generates a continuous, analytical cosine sum function
    """

    def __init__(self, data: Iterable[float], cutoff_amount: int = None, range_min: float = None,
                 range_max: float = None):
        """
        Initializes the DCT object, calculates coefficients, and optionally filters them
        :param data: The discrete set of Y-values (signal amplitude) to transform
        :param cutoff_amount: The number of largest DCT coefficients to keep. All other coefficients are zeroed out. If None, all coefficients are kept
        :param range_min: The minimum X-coordinate of the data (required for scaling)
        :param range_max: The maximum X-coordinate of the data (required for scaling)
        """
        self.range_min = range_min
        self.range_max = range_max

        # Calc DCT coefficients
        self.coefficients = sci.fft.dct(data)
        # Filter coefficients if needed
        if cutoff_amount is not None:
            indices, _ = zip(*sorted(enumerate(self.coefficients), key=lambda x: abs(x[1]), reverse=True))
            for idx in indices[cutoff_amount:]:
                self.coefficients[idx] = 0

    def __call__(self, *args, **kwargs):
        """
        Allows the class instance to be called directly as a mathematical function
        See `numpy_func` for docs
        """
        return self.numpy_func(*args, **kwargs)

    def numpy_func(self, x: float | Iterable[float], scaled: bool = False):
        """
        Evaluates the analytical cosine sum at the given point(s)

        :param x: The point(s) to evaluate the function. Can be a single float or an array of floats
        :param scaled: If True, maps  X-coordinates to the DCT index space using `range_min` and `range_max`
        :return: Y-value(s) evaluated at `x`
        """
        N = len(self.coefficients)
        if scaled:
            x = (x - self.range_min) / (self.range_max - self.range_min) * (N - 1)
        result = self.coefficients[0] / 2
        for n in range(1, N):
            result += self.coefficients[n] * np.cos((n / N * np.pi * (x + 1 / 2)))
        result *= (1 / N)
        return result
