import unittest

from networkx import is_directed_acyclic_graph

from some_pkg.domain import SchemaElement, E_Class, A_Class, Cardinality, R_Class, RSchema, RSkeleton, SkItem
from some_pkg.generators import generate_schema, generate_skeleton, between_sampler, generate_rcm, linear_gaussians_rcm, \
    generate_values_for_skeleton, average_agg, max_agg, linear_gaussian, normal_sampler
from some_pkg.model import RPath, llrsp, eqint, RVar, RDep, RCM, GroundGraph, PRCM, UndirectedRDep, flatten
from some_pkg.model import terminal_set


# TODO clean up later

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

    def test_item_classes(self):
        xx1 = E_Class('x', ['y', 'z'])
        xx2 = E_Class('x', 'y')
        xx3 = E_Class('x', A_Class('y'))
        xx4 = E_Class('x', [A_Class('y'), A_Class('z')])
        assert xx2 == xx3
        assert xx1 == xx4

    def test_EDPDE(self):
        xx1 = E_Class('x', ['y', 'z'])
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
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
        assert A_Class('Salary') in company_schema
        assert E in company_schema
        assert xx1 not in company_schema
        assert A_Class('celery') not in company_schema
        print(str(company_schema))
        print(repr(company_schema))

        e2 = E.removed({A_Class('Salary'), })
        d2 = D.removed({A_Class('dummy1'), E})
        d3 = D.removed({A_Class('dummy1'), e2})
        assert d2 == d3
        print(repr(e2))
        print(repr(d2))
        print(repr(d3))

        assert isinstance(company_schema.relateds(B), frozenset)
        assert company_schema.relateds(B) == {F, }
        assert company_schema.relateds(P) == {D, F}
        assert company_schema.relateds(E) == {D, }

    @unittest.expectedFailure
    def test_rpath_0(self):
        E, D, P, F, B = self.E, self.D, self.P, self.F, self.B
        RPath([E, D, E])

    @unittest.expectedFailure
    def test_rpath_1(self):
        E, D, P, F, B = self.E, self.D, self.P, self.F, self.B
        RPath([E, E])

    @unittest.expectedFailure
    def test_rpath_2(self):
        E, D, P, F, B = self.E, self.D, self.P, self.F, self.B
        RPath([D, D])

    @unittest.expectedFailure
    def test_rpath_3(self):
        E, D, P, F, B = self.E, self.D, self.P, self.F, self.B
        RPath([E, D, P, F, B, F, P, F])

    @unittest.expectedFailure
    def test_rpath_6(self):
        E, D, P, F, B = self.E, self.D, self.P, self.F, self.B
        RPath([E, P, F, B])

    @unittest.expectedFailure
    def test_rpath_5(self):
        E, D, P, F, B = self.E, self.D, self.P, self.F, self.B
        RPath([E, F, B])

    def test_rpath_4(self):
        E, D, P, F, B = self.E, self.D, self.P, self.F, self.B
        RPath([E])
        RPath(E)
        RPath([E, D])
        RPath([E, D, P])
        RPath([E, D, P, D])
        RPath([E, D, P, D, E])
        RPath([E, D, P, F, B])
        RPath([E, D, P, F, B, F])
        RPath([E, D, P, F, B, F, P])
        RPath([E, D, P, F, B, F, P, D])
        RPath([E, D, P, F, B, F, P, D, E])
        assert RPath([E, D, P, F, B, F, P, D, E]).joinable(RPath([E, D, P]))
        assert RPath([E, D, P, F, B, F, P, D, E]).join(RPath([E, D, P])) == RPath([E, D, P, F, B, F, P, D, E, D, P])
        assert RPath([E, D, P, F, B, F, P, D, E]).joinable(RPath(E))
        assert RPath(E).joinable(RPath(E))
        assert not RPath(E).joinable(RPath(D))
        assert RPath(E).join(RPath(E)) == RPath(E)
        assert not RPath([E, D, P, F, B, F, P]).joinable(RPath([P, F, B]))
        RPath([D])
        RPath(D)
        assert RPath(D).is_canonical
        RPath([D, P])
        assert not RPath([D, P]).is_canonical
        RPath([D, P, D])
        RPath([D, P, D, E])
        assert E == RPath([D, P, D, E]).terminal
        assert D == RPath([D, P, D, E]).base
        assert P == RPath([D, P, D, E])[1]
        assert reversed(RPath([D, P, D, E])) == RPath(tuple(reversed([D, P, D, E]))) == RPath([E, D, P, D])
        RPath([D, P, F, B])
        RPath([D, P, F, B, F])
        RPath([D, P, F, B, F, P])
        assert RPath([D, P, F, B, F, P])[2:5] == (F, B, F)
        assert RPath([D, P, F, B, F, P]).subpath(2, 5) == RPath([F, B, F])
        RPath([D, P, F, B, F, P, D])
        RPath([D, P, F, B, F, P, D, E])
        assert not RPath([D, P, F, B, F, P, D, E]).is_canonical
        assert RPath(D) == RPath([D, ])
        assert len(set([RPath([D, P, F, B, F, P, D, E]),
                        RPath([D, P, F, B, F, P, D]),
                        RPath([D, P, F, B, F, P]),
                        RPath([D, P, F, B, F, P])])) == 3
        assert all(i == j for i, j in zip(RPath([D, P, F, B, F, P, D, E]), [D, P, F, B, F, P, D, E]))
        assert llrsp(RPath([P, F, B]), RPath([P, F, B])) == 3
        assert llrsp(RPath([E, D, P, F, B]), RPath([E, D, P, D, E])) == 1
        assert llrsp(RPath([P, F, B, F, P]), RPath([P, F, B, F, P])) == 3
        assert eqint(RPath([P, F, B, F, P]), RPath([P, F, B, F, P]))
        assert eqint(RPath([E, D, P, D, E]), RPath([E, D, P, D, E, D, P, D, E]))
        assert eqint(RPath([E, D, P, D, E]), RPath([E, D, P, D, E, D, P, D, E]))
        assert RVar(RPath([E, D, P, D, E]), A_Class('Salary')) == RVar(RPath([E, D, P, D, E]), A_Class('Salary'))
        assert not RVar(RPath([E, D, P, D, E]), A_Class('Salary')).is_canonical
        assert RVar(RPath(E), A_Class('Salary')).is_canonical
        assert len(RVar(RPath(E), A_Class('Salary'))) == 1
        assert len(RVar(RPath([E, D, P, D, E]), A_Class('Salary'))) == 5
        assert len(
            {RVar(RPath([E, D, P, D, E]), A_Class('Salary')), RVar(RPath([E, D, P, D, E]), A_Class('Salary'))}) == 1


class TestSkeleton(unittest.TestCase):
    def test_skeleton(self):
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        company_schema = RSchema({E, P, B}, {D, F})

        entities = ['Paul', 'Roger', 'Quinn', 'Sally', 'Thomas',
                    'Case', 'Adapter', 'Laptop', 'Tablet', 'Smartphone',
                    'Accessories', 'Devices']
        entity_types = {'Paul': E, 'Roger': E, 'Quinn': E, 'Sally': E, 'Thomas': E,
                        'Case': P, 'Adapter': P, 'Laptop': P, 'Tablet': P, 'Smartphone': P,
                        'Accessories': B, 'Devices': B}
        skeleton = RSkeleton(True)
        p, r, q, s, t, c, a, l, ta, sm, ac, d = ents = tuple([SkItem(e, entity_types[e]) for e in entities])
        skeleton.add_entities(*ents)
        for emp, prods in ((p, {c, }), (q, {c, a, l}), (s, {l, ta}), (t, {sm, ta}), (r, {l, })):
            for prod in prods:
                skeleton.add_relationship(SkItem(emp.name + '-' + prod.name, D), {emp, prod})
        for biz, prods in ((ac, {c, a}), (d, {l, ta, sm})):
            for prod in prods:
                skeleton.add_relationship(SkItem(biz.name + '-' + prod.name, F), {biz, prod})

        assert terminal_set(skeleton, RPath([E, D, P, F, B]), p) == {ac, }
        assert terminal_set(skeleton, RPath([E, D, P, F, B]), q) == {ac, d}
        assert terminal_set(skeleton, RPath([E, D, P, D, E]), r) == {q, s}
        assert terminal_set(skeleton, RPath([E, D, P, D, E, D, P]), r) == {c, a, ta}

    def test_rcm(self):
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        company_schema = RSchema({E, P, B}, {D, F})
        deps = (RDep(RVar(E, 'Competence'), RVar(E, 'Salary')),
                RDep(RVar([E, D, P, F, B], 'Budget'), RVar(E, 'Salary')),
                RDep(RVar([P, D, E], 'Competence'), RVar(P, 'Success')),
                RDep(RVar([B, F, P], 'Success'), RVar(B, 'Revenue')),
                RDep(RVar(B, 'Revenue'), RVar(B, 'Budget')))
        rcm = RCM(company_schema, deps)
        rcm.add(deps)
        cdg = rcm.class_dependency_graph
        assert cdg.adj(A_Class('Revenue')) == {A_Class('Success'), A_Class('Budget')}
        assert cdg.ne(A_Class('Revenue')) == set()
        assert cdg.pa(A_Class('Revenue')) == {A_Class('Success'), }
        assert cdg.ch(A_Class('Revenue')) == {A_Class('Budget'), }

        entities = ['Paul', 'Roger', 'Quinn', 'Sally', 'Thomas',
                    'Case', 'Adapter', 'Laptop', 'Tablet', 'Smartphone',
                    'Accessories', 'Devices']

        entity_types = {'Paul': E, 'Roger': E, 'Quinn': E, 'Sally': E, 'Thomas': E,
                        'Case': P, 'Adapter': P, 'Laptop': P, 'Tablet': P, 'Smartphone': P,
                        'Accessories': B, 'Devices': B}
        skeleton = RSkeleton(True)
        p, r, q, s, t, c, a, l, ta, sm, ac, d = ents = tuple([SkItem(e, entity_types[e]) for e in entities])
        skeleton.add_entities(*ents)
        for emp, prods in ((p, {c, }), (q, {c, a, l}), (s, {l, ta}), (t, {sm, ta}), (r, {l, })):
            for prod in prods:
                skeleton.add_relationship(SkItem(emp.name + '-' + prod.name, D), {emp, prod})
        for biz, prods in ((ac, {c, a}), (d, {l, ta, sm})):
            for prod in prods:
                skeleton.add_relationship(SkItem(biz.name + '-' + prod.name, F), {biz, prod})

        gg = GroundGraph(rcm, skeleton)
        print(gg)

        assert rcm.max_hop == 4

        urcm = PRCM(company_schema, {UndirectedRDep(d) for d in deps})
        assert urcm.max_hop == 4
        ucdg = urcm.class_dependency_graph
        assert not ucdg.pa(A_Class('Salary'))
        assert not ucdg.ch(A_Class('Salary'))
        assert ucdg.ne(A_Class('Salary')) == {A_Class('Budget'), A_Class('Competence')}
        assert ucdg.adj(A_Class('Salary')) == {A_Class('Budget'), A_Class('Competence')}
        for d in deps:
            urcm.orient_as(d)
        cdg = urcm.class_dependency_graph
        assert cdg.adj(A_Class('Revenue')) == {A_Class('Success'), A_Class('Budget')}
        assert cdg.ne(A_Class('Revenue')) == set()
        assert cdg.pa(A_Class('Revenue')) == {A_Class('Success'), }
        assert cdg.ch(A_Class('Revenue')) == {A_Class('Budget'), }

        rcm = RCM(company_schema, [])
        assert rcm.max_hop == -1

    @unittest.expectedFailure
    def test_rcm_orient_0(self):
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        company_schema = RSchema({E, P, B}, {D, F})
        deps = (RDep(RVar(E, 'Competence'), RVar(E, 'Salary')),
                RDep(RVar([E, D, P, F, B], 'Budget'), RVar(E, 'Salary')),
                RDep(RVar([P, D, E], 'Competence'), RVar(P, 'Success')),
                RDep(RVar([B, F, P], 'Success'), RVar(B, 'Revenue')),
                RDep(RVar(B, 'Revenue'), RVar(B, 'Budget')))
        urcm = PRCM(company_schema, {UndirectedRDep(d) for d in deps})
        for d in deps:
            urcm.orient_as(d)
        for d in deps:
            urcm.orient_as(reversed(d))

    @unittest.expectedFailure
    def test_rcm_orient_1(self):
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        company_schema = RSchema({E, P, B}, {D, F})
        deps = (RDep(RVar(E, 'Competence'), RVar(E, 'Salary')),
                RDep(RVar([E, D, P, F, B], 'Budget'), RVar(E, 'Salary')),
                RDep(RVar([P, D, E], 'Competence'), RVar(P, 'Success')),
                RDep(RVar([B, F, P], 'Success'), RVar(B, 'Revenue')),
                RDep(RVar(B, 'Revenue'), RVar(B, 'Budget')))
        urcm = PRCM(company_schema, {UndirectedRDep(d) for d in deps})
        urcm.add(deps)

    @unittest.expectedFailure
    def test_rcm_orient_2(self):
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        company_schema = RSchema({E, P, B}, {D, F})
        deps = (RDep(RVar(E, 'Competence'), RVar(E, 'Salary')),
                RDep(RVar([E, D, P, F, B], 'Budget'), RVar(E, 'Salary')),
                RDep(RVar([P, D, E], 'Competence'), RVar(P, 'Success')),
                RDep(RVar([B, F, P], 'Success'), RVar(B, 'Revenue')),
                RDep(RVar(B, 'Revenue'), RVar(B, 'Budget')))
        rcm = PRCM(company_schema, deps)
        rcm.add({UndirectedRDep(d) for d in deps})

    @unittest.expectedFailure
    def test_rcm_orient_3(self):
        E = E_Class('Employee', ('Salary', 'Competence'))
        P = E_Class('Product', (A_Class('Success'),))
        B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
        D = R_Class('Develops', (A_Class('dummy1'), A_Class('dummy2')), {E: Cardinality.many, P: Cardinality.many})
        F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})
        company_schema = RSchema({E, P, B}, {D, F})
        deps = (RDep(RVar(E, 'Competence'), RVar(E, 'Salary')),
                RDep(RVar([E, D, P, F, B], 'Budget'), RVar(E, 'Salary')),
                RDep(RVar([P, D, E], 'Competence'), RVar(P, 'Success')),
                RDep(RVar([B, F, P], 'Success'), RVar(B, 'Revenue')),
                RDep(RVar(B, 'Revenue'), RVar(B, 'Budget')))
        rcm = PRCM(company_schema, deps)
        rcm.add({reversed(d) for d in deps})


class TestGenerators(unittest.TestCase):
    def test_sampler(self):
        sampler = between_sampler(10, 11)
        samples = sampler.sample(1000)
        assert len(samples) == 1000
        samples = list(sorted(samples))
        assert samples[0] == 10
        assert samples[-1] == 11

    def test_skeleton_gen(self):
        for i in range(30):
            schema = generate_schema()
            skeleton = generate_skeleton(schema)
            for R in schema.relationships:
                assert isinstance(R, R_Class)
                for E in R.entities:
                    if not R.is_many(E):
                        ents = skeleton.items(E)
                        for e in ents:
                            must_be_one = skeleton.neighbors(e, R)
                            assert len(must_be_one) <= 1

    def test_rcm_gen(self):
        s1 = between_sampler(1, 10)
        s2 = between_sampler(1, 20)
        s3 = between_sampler(1, 5)
        for i in range(200):
            schema = generate_schema()
            max_hop = s1.sample()
            max_degree = s3.sample()
            num_dependencies = s2.sample()
            rcm = generate_rcm(schema, num_dependencies, max_degree, max_hop)
            assert rcm.max_hop <= max_hop
            assert len(rcm.directed_dependencies) <= num_dependencies
            assert not rcm.undirected_dependencies
            assert is_directed_acyclic_graph(rcm.class_dependency_graph.as_networkx_dag())
            effects = {e for c, e in rcm.directed_dependencies}
            assert all(len(rcm.pa(e)) <= max_degree for e in effects)
            for d in rcm.directed_dependencies:
                print(d)

            print()

    def test_data_gen(self):
        for i in range(10):
            schema = generate_schema()
            rcm = generate_rcm(schema, 20, 5, 4)
            skeleton = generate_skeleton(schema)
            lg_rcm = linear_gaussians_rcm(rcm)
            generate_values_for_skeleton(lg_rcm, skeleton)
            causes = {c for c, e in rcm.directed_dependencies}
            one_cause = next(iter(causes))
            causes_of_the_base = list(filter(lambda c: c.base == one_cause.base, causes))
            data_frame = flatten(skeleton, causes_of_the_base)



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
