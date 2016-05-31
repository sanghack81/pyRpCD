import multiprocessing
import unittest
from itertools import combinations

import numpy as np

from pyrcds.__rci import normalize, multiply, SetKernelRCITester
from pyrcds.domain import generate_skeleton
from pyrcds.model import generate_values_for_skeleton, ParamRCM, RPath, RVar
from pyrcds.utils import linear_gaussian, average_agg, normal_sampler
from tests.testing_utils import company_rcm, company_schema


class TestCI(unittest.TestCase):
    def test_normalize(self):
        for _ in range(10):
            x = abs(np.random.randn(10, 10)) + 1.0e-6
            x += x.transpose()
            x = normalize(x)
            for v in np.diag(x):
                assert abs(v - 1.0) <= 1.0e-6

        assert normalize(None) is None

    def test_multiply(self):
        x = abs(np.random.randn(10, 10)) + 1.0e-6
        y = abs(np.random.randn(10, 10)) + 1.0e-6
        z = abs(np.random.randn(10, 10)) + 1.0e-6

        assert np.allclose(multiply(x, y, z), x * y * z)
        assert np.allclose(multiply(x, y), x * y)
        assert np.allclose(multiply(x), x)
        assert multiply() is None

    def test_ci_company_domain(self):
        schema, rcm = company_schema(), company_rcm()
        skeleton = generate_skeleton(schema)

        functions = dict()
        effects = {RVar(RPath(rcm.schema.item_class_of(attr)), attr) for attr in rcm.schema.attrs}

        for e in effects:
            parameters = {cause: 1.0 for cause in rcm.pa(e)}
            # with non-zero mean
            functions[e] = linear_gaussian(parameters, average_agg(), normal_sampler(1, 0.3))

        lg_rcm = ParamRCM(rcm.schema, rcm.directed_dependencies, functions)

        generate_values_for_skeleton(lg_rcm, skeleton)

        n_cpu = multiprocessing.cpu_count()
        B = n_cpu
        while B < 16:
            B += n_cpu

        tester = SetKernelRCITester(skeleton, maxed=300, alpha=0.05, n_jobs=1, B=1)
        print('degree: {}'.format(lg_rcm.degree))
        for cond_size in range(lg_rcm.degree):
            print('with cond size: {}'.format(cond_size))
            for d0 in lg_rcm.directed_dependencies:
                for d in ((d0, reversed(d0)) if cond_size > 0 else (d0,)):
                    conds = lg_rcm.adj(d.effect) - {d.cause}
                    for cond in combinations(conds, cond_size):
                        print('testing {} _||_ {} | {}'.format(d.cause, d.effect, cond))
                        ci_result = tester.ci_test(d.cause, d.effect, cond)
                        if ci_result.ci:
                            print("wrong:  {}: {}".format(d, ci_result))
                        else:
                            print("passed: {}: {}".format(d, ci_result))
                    print()


if __name__ == '__main__':
    unittest.main()
