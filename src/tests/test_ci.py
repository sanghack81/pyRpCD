import unittest

import numpy as np
from numpy.random import randn

from pyrcds.__rci import normalize, multiply, SetKernelRCITester
from pyrcds.domain import generate_skeleton
from pyrcds.model import generate_values_for_skeleton, ParamRCM, RPath, RVar
from pyrcds.utils import linear_gaussian, average_agg, normal_sampler
from tests.testing_utils import company_rcm, company_schema


class TestCI(unittest.TestCase):
    def test_normalize(self):
        for _ in range(10):
            x = abs(np.random.randn(10, 10)) + (1.0e-6)
            x += x.transpose()
            x = normalize(x)
            for v in np.diag(x):
                assert abs(v - 1.0) <= 1.0e-6

        assert normalize(None) is None

    def test_multiply(self):
        x = abs(np.random.randn(10, 10)) + (1.0e-6)
        y = abs(np.random.randn(10, 10)) + (1.0e-6)
        z = abs(np.random.randn(10, 10)) + (1.0e-6)

        assert all(((multiply(x, y, z) - (x * y * z)) < 1.0e-6).flatten())
        assert all(((multiply(x, y) - (x * y)) < 1.0e-6).flatten())
        assert all(((multiply(x) - x) < 1.0e-6).flatten())
        assert multiply() is None

    def test_ci_company_domain(self):
        schema, rcm = company_schema(), company_rcm()

        for i in range(10):
            print("iteration {}".format(i))
            skeleton = generate_skeleton(schema)

            functions = dict()
            effects = {RVar(RPath(rcm.schema.item_class_of(attr)), attr) for attr in rcm.schema.attrs}

            for e in effects:
                parameters = {cause: 1.0 for cause in rcm.pa(e)}
                functions[e] = linear_gaussian(parameters, average_agg(), normal_sampler(0, 0.1))

            lg_rcm = ParamRCM(rcm.schema, rcm.directed_dependencies, functions)

            generate_values_for_skeleton(lg_rcm, skeleton)

            tester = SetKernelRCITester(skeleton, alpha=0.05, n_jobs=4, B=12, timeout_linprog=60, timeout_for_quick=15)
            for d in lg_rcm.directed_dependencies:
                ci_result = tester.ci_test(d.cause, d.effect)
                if ci_result.ci:
                    print("wrong:  {}: {}".format(d, ci_result))
                else:
                    print("passed: {}: {}".format(d, ci_result))
            print()


if __name__ == '__main__':
    unittest.main()
