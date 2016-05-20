import unittest

from pyrcds.utils import between_sampler, average_agg, max_agg, normal_sampler, linear_gaussian


class TestGenerators(unittest.TestCase):
    def test_sampler(self):
        sampler = between_sampler(10, 11)
        samples = sampler.sample(1000)
        assert len(samples) == 1000
        samples = list(sorted(samples))
        assert samples[0] == 10
        assert samples[-1] == 11

    def test_aggregators(self):
        agg = average_agg(1.0)
        assert 2.0 == agg([1, 2, 3])
        assert 1.0 == agg(set())

        agg = max_agg(1.0)
        assert 3.0 == agg([1, 2, 3])
        assert 1.0 == agg(set())

    def test_linear_gaussian(self):
        lgm = linear_gaussian({'x': 2.0, 'y': 1.0}, average_agg(), normal_sampler(0.0, 0.0))
        values = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        cause_attrs = {'x': ['a', 'b', 'c'], 'y': ['c']}

        v = lgm(values, cause_attrs)
        assert v == 2.0 * (sum(values.values()) / len(values)) + 1.0 * 3.0

    def test_linear_max_gaussian(self):
        lgm = linear_gaussian({'x': 2.0, 'y': 1.0}, max_agg(), normal_sampler(1.0, 0.0))
        values = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        cause_attrs = {'x': ['a', 'b', 'c'], 'y': ['c']}

        v = lgm(values, cause_attrs)
        assert v == 2.0 * 3.0 + 1.0 * 3.0 + 1.0


if __name__ == '__main__':
    unittest.main()
