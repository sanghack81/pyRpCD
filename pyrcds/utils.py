import types
from itertools import groupby, chain

import numpy as np
from numpy import median
from numpy.random import randint, randn


def unions(sets):
    if isinstance(sets, types.GeneratorType):
        return set(chain(*list(sets)))
    else:
        return set(chain(*sets))


def group_by(xs, keyfunc):
    """A generator of tuples of a key and its group as a `list`"""
    return ((k, list(g)) for k, g in groupby(sorted(xs, key=keyfunc), key=keyfunc))


def safe_iter(iterable):
    """Iterator (generator) that can skip removed items."""
    copied = list(iterable)
    for y in copied:
        if y in iterable:
            yield y


class between_sampler:
    """Random integer sampler which returns an `int` between given `min_inclusive` and `max_inclusive`."""

    def __init__(self, min_inclusive, max_inclusive):
        assert min_inclusive <= max_inclusive
        self.m = min_inclusive
        self.M = max_inclusive

    def sample(self, size=None):
        if size is None:
            return randint(self.m, self.M + 1)
        else:
            return randint(self.m, self.M + 1, size=size)


class normal_sampler:
    def __init__(self, mu=0.0, sd=1.0):
        self.mu = mu
        self.sd = sd

    def sample(self):
        return self.sd * randn() + self.mu


def average_agg(default=0.0):
    """A function that returns the average of a given input or `default` value if the given input is empty."""

    return lambda items: (sum(items) / len(items)) if len(items) > 0 else default


def max_agg(default=0.0):
    """A function that returns the maximum value of a given input
     or default value if the given input is empty.
     """

    return lambda items: max(items) if len(items) > 0 else default


def linear_gaussian(parameters: dict, aggregator, error):
    """A linear model with an additive Gaussian noise.

    Parameters
    ----------
    parameters : parameters for the linear model. e.g., parameter = parameters[cause_rvar]
    aggregator: a function that maps multiple values to a single value.
    error: additive noise distribution, err = error.sample()

    Returns
    -------
    function
        a linear Gaussian model with given parameters and an aggregator
    """

    def func(values, cause_item_attrs):
        value = 0

        for rvar in sorted(parameters.keys()):
            item_attr_values = [values[item_attr] for item_attr in cause_item_attrs[rvar]]
            value += parameters[rvar] * aggregator(item_attr_values)

        return value + error.sample()

    return func


def xors(parameters):
    """A function that xor-s given values."""

    def func(values, cause_item_attrs):
        value = 0 if parameters else randint(2)

        for rvar in sorted(parameters.keys()):
            for item_attr in cause_item_attrs[rvar]:
                value ^= values[item_attr]

        return value

    return func


def median_except_diag(D, exclude_inf=True, default=1):
    """A median value of a matrix except diagonal elements.

    Parameters
    ----------
    D : matrix-like
    exclude_inf : bool
        whether to exclude infinity
    default : int
        default return value if there is no value to compute median
    """
    return stat_except_diag(D, exclude_inf, default, median)


def mean_except_diag(D, exclude_inf=True, default=1):
    """A mean value of a matrix except diagonal elements.

    Parameters
    ----------
    D : array_like
        Array to be measured
    exclude_inf : bool
        whether to exclude infinity
    default : int
        default return value if there is no value to compute mean
    """
    return stat_except_diag(D, exclude_inf, default, np.mean)


def stat_except_diag(D, exclude_inf=True, default=1, func=median):
    if D.ndim != 2:
        raise TypeError('not a matrix')
    if D.shape[0] != D.shape[1]:
        raise TypeError('not a square matrix')
    if len(D) <= 1:
        raise ValueError('No non-diagonal element')

    lower = D[np.tri(len(D), k=-1, dtype=bool)]
    upper = D.transpose()[np.tri(len(D), k=-1, dtype=bool)]
    non_diagonal = np.concatenate((lower, upper))
    if exclude_inf:
        non_diagonal = non_diagonal[non_diagonal != float('inf')]

    if len(non_diagonal):
        return func(non_diagonal)
    else:
        return default
