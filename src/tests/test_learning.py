import unittest

from some_pkg.domain import *
from some_pkg.learn.learning import enumerate_rpaths, enumerate_rdeps
from some_pkg.model import is_valid_relational_path, RDep, RVar, RCM


class TestLearning(unittest.TestCase):
    def test_enumerate_rpaths(self):
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', tuple(), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        company_schema = RSchema({E, P, B}, {D, F})

        entities = ['Paul', 'Roger', 'Quinn', 'Sally', 'Thomas',
                    'Case', 'Adapter', 'Laptop', 'Tablet', 'Smartphone',
                    'Accessories', 'Devices']
        entity_types = {'Paul': E, 'Roger': E, 'Quinn': E, 'Sally': E, 'Thomas': E,
                        'Case': P, 'Adapter': P, 'Laptop': P, 'Tablet': P, 'Smartphone': P,
                        'Accessories': B, 'Devices': B}

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


if __name__ == '__main__':
    unittest.main()
