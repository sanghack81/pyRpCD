import unittest

import numpy as np
from joblib import Parallel, delayed
from networkx import is_directed_acyclic_graph

from pyrcds.domain import A_Class, RSkeleton, SkItem, generate_schema, generate_skeleton
from pyrcds.model import RPath, llrsp, eqint, RVar, terminal_set, PRCM, UndirectedRDep, RCM, GroundGraph, generate_rcm, \
    canonical_rvars, linear_gaussians_rcm, generate_values_for_skeleton, flatten, generate_rpath, is_valid_rpath
from pyrcds.utils import between_sampler
from tests.testing_utils import EPBDF, company_schema, company_rcm, company_skeleton, company_deps


class TestModel(unittest.TestCase):
    @unittest.expectedFailure
    def test_rpath_0(self):
        E, P, B, D, F = EPBDF()
        RPath([E, D, E])

    @unittest.expectedFailure
    def test_rpath_1(self):
        E, P, B, D, F = EPBDF()
        RPath([E, E])

    @unittest.expectedFailure
    def test_rpath_2(self):
        E, P, B, D, F = EPBDF()
        RPath([D, D])

    @unittest.expectedFailure
    def test_rpath_3(self):
        E, P, B, D, F = EPBDF()
        RPath([E, D, P, F, B, F, P, F])

    @unittest.expectedFailure
    def test_rpath_6(self):
        E, P, B, D, F = EPBDF()
        RPath([E, P, F, B])

    @unittest.expectedFailure
    def test_rpath_5(self):
        E, P, B, D, F = EPBDF()
        RPath([E, F, B])

    def test_rpath_4(self):
        E, P, B, D, F = EPBDF()
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
        assert RPath([D, P, F, B, F, P])[2:5] == RPath((F, B, F, P))
        assert RPath([D, P, F, B, F, P]).subpath(2, 5) == RPath([F, B, F])
        RPath([D, P, F, B, F, P, D])
        RPath([D, P, F, B, F, P, D, E])
        assert not RPath([D, P, F, B, F, P, D, E]).is_canonical
        assert RPath(D) == RPath([D, ])
        assert len({RPath([D, P, F, B, F, P, D, E]), RPath([D, P, F, B, F, P, D]), RPath([D, P, F, B, F, P]),
                    RPath([D, P, F, B, F, P])}) == 3
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

    def test_skeleton(self):
        E, P, B, D, F = EPBDF()
        entities = ['Paul', 'Roger', 'Quinn', 'Sally', 'Thomas',
                    'Case', 'Adapter', 'Laptop', 'Tablet', 'Smartphone',
                    'Accessories', 'Devices']

        entity_types = {'Paul': E, 'Roger': E, 'Quinn': E, 'Sally': E, 'Thomas': E,
                        'Case': P, 'Adapter': P, 'Laptop': P, 'Tablet': P, 'Smartphone': P,
                        'Accessories': B, 'Devices': B}
        skeleton = RSkeleton(company_schema(), True)
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

    def test_sub(self):
        E, P, B, D, F = EPBDF()
        p = RPath([E, D, P, F, B])
        assert p[:] == RPath([E, D, P, F, B])
        assert p[:2] == RPath([E, D, P])
        assert p[1:] == RPath([D, P, F, B])
        assert p[1:2] == RPath([D, P])
        assert p[::-1] == RPath([B, F, P, D, E])
        assert p[:2:-1] == RPath([P, D, E])
        assert p[1::-1] == RPath([B, F, P, D])
        assert p[1:2:-1] == RPath([P, D])

    def test_rcm(self):
        deps = company_deps()
        rcm = company_rcm()
        urcm = PRCM(company_schema(), {UndirectedRDep(d) for d in rcm.directed_dependencies})
        skeleton = company_skeleton()

        gg = GroundGraph(rcm, skeleton)
        directed_uts = gg.unshielded_triples()
        assert len(directed_uts) == 57
        dag_gg = gg.as_networkx_dag()

        # TODO any test?

        ugg = GroundGraph(urcm, skeleton)
        undirected_uts = ugg.unshielded_triples()
        assert directed_uts == undirected_uts

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

    def test_crv(self):
        rvars = canonical_rvars(company_schema())
        E, P, B, D, F = EPBDF()
        assert rvars == {RVar(RPath(E), 'Competence'), RVar(RPath(E), 'Salary'), RVar(RPath(B), 'Revenue'),
                         RVar(RPath(B), 'Budget'), RVar(RPath(P), 'Success')}

    @unittest.expectedFailure
    def test_rcm_orient_0(self):
        deps = company_deps()
        urcm = PRCM(company_schema, {UndirectedRDep(d) for d in deps})
        for d in deps:
            urcm.orient_as(d)
        for d in deps:
            urcm.orient_as(reversed(d))

    @unittest.expectedFailure
    def test_rcm_orient_1(self):
        deps = company_deps()
        urcm = PRCM(company_schema, {UndirectedRDep(d) for d in deps})
        urcm.add(deps)

    @unittest.expectedFailure
    def test_rcm_orient_2(self):
        deps = company_deps()
        rcm = company_rcm()
        rcm.add({UndirectedRDep(d) for d in deps})

    @unittest.expectedFailure
    def test_rcm_orient_3(self):
        deps = company_deps()
        rcm = company_rcm()
        rcm.add({reversed(d) for d in deps})

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

    def test_generate_rpaths(self):
        for _ in range(100):
            schema = generate_schema()
            for __ in range(100):
                rpath = generate_rpath(schema, length=np.random.randint(1, 15))
                assert is_valid_rpath([i for i in rpath])

    def test_data_gen(self):
        n = 20
        seeds = [np.random.randint(np.iinfo(np.int32).max) for _ in range(n)]
        Parallel(-1)(delayed(_test_gen_inner)(seeds[i]) for i in range(n))


def _test_gen_inner(seed):
    np.random.seed(seed)

    schema = generate_schema()
    rcm = generate_rcm(schema, 20, 5, 4)
    skeleton = generate_skeleton(schema)
    lg_rcm = linear_gaussians_rcm(rcm)
    generate_values_for_skeleton(lg_rcm, skeleton)
    causes = {c for c, e in rcm.directed_dependencies}
    one_cause = next(iter(causes))
    causes_of_the_base = list(filter(lambda c: c.base == one_cause.base, causes))
    data_frame0 = flatten(skeleton, causes_of_the_base, False, False)
    data_frame1 = flatten(skeleton, causes_of_the_base, True, True)
    data_frame2 = flatten(skeleton, causes_of_the_base, False, True)
    data_frame3 = flatten(skeleton, causes_of_the_base, True, False)
    # base items
    assert all((data_frame3[:, 0] == data_frame1[:, 0]).flatten())
    assert all((data_frame1[:, 1] == data_frame2[:, 0]).flatten())
    assert all((data_frame0[:, 0] == data_frame3[:, 1]).flatten())
    assert sorted(sorted(val for item, val in ivs) for ivs in data_frame0[:, 0]) == sorted(
        sorted(v) for v in data_frame1[:, 1])


if __name__ == '__main__':
    unittest.main()
