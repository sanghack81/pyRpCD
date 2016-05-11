import unittest

from pyrcds.domain import E_Class, A_Class, Cardinality, R_Class, RSchema
from pyrcds.model import RCM, RDep, RVar
from pyrcds.utils import generate_skeleton, linear_gaussians_rcm, \
    generate_values_for_skeleton


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.E = E_Class('Employee', ('Salary', 'Competence'))
        E = self.E
        self.P = E_Class('Product', (A_Class('Success'),))
        P = self.P
        self.B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        B = self.B
        self.D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
        D = self.D
        self.F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        F = self.F

    def test_ci_company_domain(self):
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        schema = RSchema({E, P, B}, {D, F})

        deps = (RDep(RVar(E, 'Competence'), RVar(E, 'Salary')),
                RDep(RVar([E, D, P, F, B], 'Budget'), RVar(E, 'Salary')),
                RDep(RVar([P, D, E], 'Competence'), RVar(P, 'Success')),
                RDep(RVar([B, F, P], 'Success'), RVar(B, 'Revenue')),
                RDep(RVar(B, 'Revenue'), RVar(B, 'Budget')))

        rcm = RCM(schema, deps)

        for i in range(10):
            skeleton = generate_skeleton(schema)
            lg_rcm = linear_gaussians_rcm(rcm)
            generate_values_for_skeleton(lg_rcm, skeleton)
            causes = {c for c, e in rcm.directed_dependencies}
            one_cause = next(iter(causes))
            causes_of_the_base = list(filter(lambda c: c.base == one_cause.base, causes))

            # def test_ci_random_schema(self):
            #     E = E_Class('Employee', ('Salary', 'Competence'))
            #     P = E_Class('Product', (A_Class('Success'),))
            #     B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
            #     D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
            #     F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
            #     schema = RSchema({E, P, B}, {D, F})
            #
            #     for i in range(10):
            #         schema = generate_schema()
            #         rcm = generate_rcm(schema, 20, 5, 4)
            #         skeleton = generate_skeleton(schema)
            #         lg_rcm = linear_gaussians_rcm(rcm)
            #         generate_values_for_skeleton(lg_rcm, skeleton)
            #         causes = {c for c, e in rcm.directed_dependencies}
            #         one_cause = next(iter(causes))
            #         causes_of_the_base = list(filter(lambda c: c.base == one_cause.base, causes))


if __name__ == '__main__':
    unittest.main()
