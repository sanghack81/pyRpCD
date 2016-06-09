import unittest
from itertools import combinations

import numpy as np

from pyrcds._rci import multiplys, SetKernelRCITester, GraphKernelRCITester
from pyrcds._spaces import normalize_by_diag
from pyrcds.domain import generate_skeleton, repeat_skeleton
from pyrcds.model import generate_values_for_skeleton, ParamRCM, RPath, RVar
from pyrcds.tests.testing_utils import company_rcm, company_schema
from pyrcds.utils import linear_gaussian, average_agg, normal_sampler


class TestCI(unittest.TestCase):
    def test_normalize(self):
        for _ in range(10):
            x = abs(np.random.randn(10, 10)) + 1.0e-6
            x += x.transpose()
            x = normalize_by_diag(x)
            for v in np.diag(x):
                assert abs(v - 1.0) <= 1.0e-6

        assert normalize_by_diag(None) is None

    def test_multiply(self):
        x = abs(np.random.randn(10, 10)) + 1.0e-6
        y = abs(np.random.randn(10, 10)) + 1.0e-6
        z = abs(np.random.randn(10, 10)) + 1.0e-6

        assert np.allclose(multiplys(x, y, z), x * y * z)
        assert np.allclose(multiplys(x, y), x * y)
        assert np.allclose(multiplys(x), x)
        assert multiplys() is None

    # @unittest.skip('time consuming, non-test')
    def test_ci_company_domain(self):
        np.random.seed(0)
        options = {'n_jobs': -1, 'alpha': 0.05, 'B': 8, 'b': 800, 'M': 10000}
        schema, rcm = company_schema(), company_rcm()
        functions = dict()
        effects = {RVar(RPath(rcm.schema.item_class_of(attr)), attr) for attr in rcm.schema.attrs}

        skeleton = generate_skeleton(schema, 80)
        skeleton = repeat_skeleton(skeleton, 5)

        for e in effects:
            parameters = {cause: 1.0 for cause in rcm.pa(e)}
            # with non-zero mean
            functions[e] = linear_gaussian(parameters, average_agg(), normal_sampler(1, 0.5))

        lg_rcm = ParamRCM(rcm.schema, rcm.directed_dependencies, functions)

        generate_values_for_skeleton(lg_rcm, skeleton)

        tester_gk = GraphKernelRCITester(skeleton, **options)
        tester_haus = SetKernelRCITester(skeleton, **options)

        print('degree: {}'.format(lg_rcm.degree))
        for cond_size in range(lg_rcm.degree):
            print('with cond size: {}'.format(cond_size))
            for d0 in lg_rcm.directed_dependencies:
                for d in ((d0, reversed(d0)) if cond_size > 0 else (d0,)):
                    conds = lg_rcm.adj(d.effect) - {d.cause}
                    for cond in combinations(conds, cond_size):
                        print('testing {} _||_ {} | {}'.format(d.cause, d.effect, cond))
                        ci_result = tester_gk.ci_test(d.cause, d.effect, cond)
                        if ci_result.ci:
                            print("wrong gk:  {}: {}".format(d, ci_result.p))
                        else:
                            print("passed gk: {}: {}".format(d, ci_result.p))
                        ci_result = tester_haus.ci_test(d.cause, d.effect, cond)
                        if ci_result.ci:
                            print("wrong haus:  {}: {}".format(d, ci_result.p))
                        else:
                            print("passed haus: {}: {}".format(d, ci_result.p))
                    print()


if __name__ == '__main__':
    unittest.main()
