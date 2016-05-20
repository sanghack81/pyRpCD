from numpy.random import randint, randn


def group_by(xs, keyfunc):
    from itertools import groupby
    return ((k, list(g)) for k, g in groupby(sorted(xs, key=keyfunc), key=keyfunc))


#
class between_sampler:
    def __init__(self, min_inclusive, max_inclusive):
        assert min_inclusive <= max_inclusive
        self.m = min_inclusive
        self.M = max_inclusive

    def sample(self, size=None):
        if size is None:
            return randint(self.m, self.M + 1)
        else:
            return randint(self.m, self.M + 1, size=size).tolist()


#
#
class normal_sampler:
    def __init__(self, mu=0.0, sd=1.0):
        self.mu = mu
        self.sd = sd

    def sample(self):
        return self.sd * randn() + self.mu


# sample() = sample(1)
# sample(n)


# Erdos-Renyi


def average_agg(default=0.0):
    def func(items):
        if len(items) > 0:
            return sum(items) / len(items)
        else:
            return default

    return func


def max_agg(default=0.0):
    def func(items):
        if len(items) > 0:
            return max(items)
        else:
            return default

    return func

# randomly generated parameters
# linear additive Gaussian


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
