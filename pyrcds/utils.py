from itertools import groupby

import numpy as np
from numpy import median
from numpy.random import randint, randn


def group_by(xs, keyfunc):
    """Returns a generator of a 2-tuple with key and list-ed group"""
    return ((k, list(g)) for k, g in groupby(sorted(xs, key=keyfunc), key=keyfunc))


def safe_iter(iterable):
    copied = list(iterable)
    for y in copied:
        if y in iterable:
            yield y


class between_sampler:
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
    """Returns a function that returns the average of a given input or default value if the given input is empty."""

    def func(items):
        return (sum(items) / len(items)) if len(items) > 0 else default

    return func


def max_agg(default=0.0):
    """Returns a function that returns the maximum value of a given input or default value if the given input is empty."""

    def func(items):
        return max(items) if len(items) > 0 else default

    return func


def linear_gaussian(parameters: dict, aggregator, error):
    """
    a linear model with an additive Gaussian noise

    :param parameters: parameters for linear model. parameter = parameters[cause_rvar]
    :param aggregator: a function that maps multiple values to a single value.
    :param error: additive noise distribution, err = error.sample()
    :return: a function that can be used in a parametrized RCM
    """

    def func(values, cause_item_attrs):
        value = 0
        for rvar in parameters:
            item_attr_values = [values[item_attr] for item_attr in cause_item_attrs[rvar]]
            value += parameters[rvar] * aggregator(item_attr_values)
        return value + error.sample()

    return func


def median_except_diag(D, exclude_inf=True, default=1):
    return stat_except_diag(D, exclude_inf, default, median)


def mean_except_diag(D, exclude_inf=True, default=1):
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
