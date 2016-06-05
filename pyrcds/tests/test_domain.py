import unittest

import numpy as np

from pyrcds.domain import SchemaElement, E_Class, A_Class, Cardinality, R_Class, RSchema, generate_schema, \
    generate_skeleton, ImmutableRSkeleton
from pyrcds.tests.testing_utils import company_skeleton, EPBDF


class TestSchemaElement(unittest.TestCase):
    @unittest.expectedFailure
    def test_SchemaElement_fail_none(self):
        self.expeSchemaElement(SchemaElement())

    @unittest.expectedFailure
    def test_SchemaElement_fail_empty(self):
        self.expeSchemaElement(SchemaElement(''))

    @unittest.expectedFailure
    def test_SchemaElement_fail_assign_name(self):
        x = SchemaElement('asd')
        x.name = 'sdf'

    def test_SchemaElement(self):
        x, y = SchemaElement('asd'), SchemaElement('asd')
        self.assertEqual(len({x, y}), 1)
        self.assertEqual(x.name, 'asd')

        x, y = SchemaElement('asd'), SchemaElement('abc')
        self.assertEqual(len({x, y}), 2)


class TestSchema(unittest.TestCase):
    def test_item_classes(self):
        xx1 = E_Class('x', ['y', 'z'])
        xx2 = E_Class('x', 'y')
        xx3 = E_Class('x', A_Class('y'))
        xx4 = E_Class('x', [A_Class('y'), A_Class('z')])
        assert xx2 == xx3
        assert xx1 == xx4

    def test_EDPDE(self):
        xx1 = E_Class('x', ['y', 'z'])
        E, P, B, _, F = EPBDF()
        D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})

        assert F[P] == Cardinality.one
        assert F[B] == Cardinality.many
        assert F.is_many(B)
        assert not F.is_many(P)
        assert P in F
        assert B in F
        assert P in D
        assert E in D
        assert E not in F
        assert D not in F
        assert F not in F
        company_schema = RSchema({E, P, B}, {D, F})
        assert company_schema.item_class_of(A_Class('Salary')) == E
        assert company_schema.item_class_of(A_Class('Revenue')) == B
        assert A_Class('Salary') in company_schema
        assert E in company_schema
        assert xx1 not in company_schema
        assert A_Class('celery') not in company_schema
        assert str(company_schema) == 'RSchema(BizUnit, Develops, Employee, Funds, Product)'
        # print(repr(company_schema))
        assert repr(
            company_schema) == "RSchema(Entity classes: [BizUnit(Budget, Revenue), Employee(Competence, Salary), Product(Success)], Relationship classes: [Develops(dummy1, dummy2, {Employee: many, Product: many}), Funds((), {BizUnit: many, Product: one})])"

        # e2 = E.removed({A_Class('Salary'), })
        # d2 = D.removed({A_Class('dummy1'), E})
        # d3 = D.removed({A_Class('dummy1'), e2})
        # assert d2 == d3

        assert isinstance(company_schema.relateds(B), frozenset)
        assert company_schema.relateds(B) == {F, }
        assert company_schema.relateds(P) == {D, F}
        assert company_schema.relateds(E) == {D, }


class TestSkeleton(unittest.TestCase):
    def test_skeleton(self):
        company_skeleton()

    def test_skeleton_gen(self):
        for i in range(30):
            schema = generate_schema()
            skeleton = generate_skeleton(schema)
            iskeleton = ImmutableRSkeleton(skeleton)
            for R in schema.relationships:
                assert isinstance(R, R_Class)
                for E in R.entities:
                    if not R.is_many(E):
                        ents = skeleton.items(E)
                        assert iskeleton.items(E) == ents
                        for e in ents:
                            must_be_one = skeleton.neighbors(e, R)
                            assert len(iskeleton.neighbors(e, R)) <= 1
                            assert len(must_be_one) <= 1
                            if 'key' not in e:
                                assert skeleton[(e, 'key')] is None
                            v = np.random.randint(100)
                            skeleton[(e, 'key')] = v
                            assert 'key' in e
                            assert v == skeleton[(e, 'key')]
                            assert v == e['key']
                            try:
                                iskeleton[(e, 'key')] = 100
                                assert False
                            except AssertionError:
                                pass


if __name__ == '__main__':
    unittest.main()
