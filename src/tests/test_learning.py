import unittest

from pyrcds.domain import *
from pyrcds.learning import enumerate_rpaths, enumerate_rdeps
from pyrcds.model import is_valid_relational_path, RDep, RVar, RCM
from pyrcds.utils import generate_skeleton, linear_gaussians_rcm, generate_values_for_skeleton


class TestLearning(unittest.TestCase):
    def test_something(self):
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', tuple(), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        company_schema = RSchema({E, P, B}, {D, F})

        deps = (RDep(RVar(E, 'Competence'), RVar(E, 'Salary')),
                RDep(RVar([E, D, P, F, B], 'Budget'), RVar(E, 'Salary')),
                RDep(RVar([P, D, E], 'Competence'), RVar(P, 'Success')),
                RDep(RVar([B, F, P], 'Success'), RVar(B, 'Revenue')),
                RDep(RVar(B, 'Revenue'), RVar(B, 'Budget')))

        rcm = RCM(company_schema, deps)

        rpaths = set(enumerate_rpaths(company_schema, 4))
        assert len(rpaths) == 43
        for rpath in enumerate_rpaths(company_schema, 4):
            assert is_valid_relational_path(rpath)

        assert rcm.directed_dependencies <= set(enumerate_rdeps(company_schema, rcm.max_hop))
        assert 22 == len(set(enumerate_rdeps(company_schema, 4)))
        assert 162 == len(set(enumerate_rdeps(company_schema, 16)))

        skeleton = generate_skeleton(company_schema)
        # draw_skeleton(skeleton)

        lg_rcm = linear_gaussians_rcm(rcm)
        generate_values_for_skeleton(lg_rcm, skeleton)

        # divide by median?
        # # TODO kernel not distance!
        # tester = SetKernelRCITester(skeleton, n_job=10)
        # learner = PracticalLearner(company_schema, rcm.max_hop, skeleton, tester)
        # logger = logging.getLogger(PracticalLearner.__module__)
        # logger2 = logging.getLogger(PracticalLearner.__module__ + '.' + PracticalLearner.__name__)
        # logger.setLevel(logging.INFO)
        # sh = logging.StreamHandler(sys.stdout)
        # sh.setLevel(logging.INFO)
        # logger.addHandler(sh)
        # print(logger.getEffectiveLevel())
        # logger.info('working?')
        # print('above working?')
        # logger.error('working error?')
        # logger.warn('working warn?')
        # logger2.warn('2 working warn?')
        # logger2.info('2working info?')
        # undirecteds = learner.phase_I()
        # true_undies = {UndirectedRDep(d) for d in rcm.directed_dependencies}
        # recall = len(true_undies & undirecteds) / len(true_undies)
        # prec = len(true_undies & undirecteds) / len(undirecteds)
        # print((prec, recall))


if __name__ == '__main__':
    unittest.main()
