import unittest
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed

from pyrcds.domain import generate_schema, R_Class, E_Class, Cardinality
from pyrcds.graphs import PDAG
from pyrcds.model import generate_rcm, RPath, RVar, SymTriple
from pyrcds.rcds import canonical_unshielded_triples, enumerate_rpaths, enumerate_rvars, interner, extend, \
    enumerate_rdeps, intersectible, UnvisitedQueue, AbstractGroundGraph, sound_rules, completes, d_separated, \
    co_intersectible
from pyrcds.tests import company_rcm, company_schema, EPBDF


class TestRCDs(unittest.TestCase):
    def test_enumerate_rpaths(self):
        M = company_rcm()
        assert {d.cause.rpath for d in M.directed_dependencies} <= set(enumerate_rpaths(M.schema, M.max_hop))

        schema = company_schema()
        assert len(set(enumerate_rpaths(schema, 0))) == len(schema.item_classes)

        expected = iter([5, 13, 22, 32, 43, 55, 69, 85, 103, 123])
        for i in range(10):
            x = list(enumerate_rpaths(schema, i))
            y = set(x)
            assert len(x) == len(y)
            assert len(y) == next(expected)

    def test_enumerate_rvars(self):
        schema = company_schema()
        for i in range(10):
            l = list(enumerate_rvars(schema, i))
            s = set(l)
            assert len(s) == len(l)
            assert len(l) == sum(
                len(p.terminal.attrs) for p in enumerate_rpaths(schema, i))

    def test_interner(self):
        xx = interner()
        x = (1, 2, 3)
        y = (1, 2, 3)
        assert id(x) != id(y)
        z1 = xx[x]
        z2 = xx[y]
        assert id(z1) == id(z2) == id(x)

    def test_enumerate_rdeps(self):
        schema = company_schema()
        rcm = company_rcm()
        assert rcm.directed_dependencies <= set(enumerate_rdeps(schema, rcm.max_hop))
        expected = [4, 4, 12, 12, 22, 22, 34, 34, 48, 48]
        for h in range(10):
            assert expected[h] == len(set(enumerate_rdeps(schema, h)))

    def test_extend(self):
        E, P, B, D, F = EPBDF()
        actual = set(extend(RPath([E, D, P, F, B]), RPath([B, F, P, D, E])))
        expected = {RPath([E, D, P, F, B, F, P, D, E]),
                    RPath([E, D, P, D, E]),
                    RPath([E])}
        assert actual == expected
        actual = set(extend(RPath([E, D, P, F, B, F, P]), RPath([P, F, B])))
        expected = {RPath([E, D, P, F, B]), }
        assert actual == expected

    def test_intersectible(self):
        E, P, B, D, F = EPBDF()

        assert intersectible(RPath([E, D, P, F, B, F, P, D, E]), RPath([E, D, P, D, E]))
        assert not intersectible(RPath([E, D, P, F, B, F, P]), RPath([E, D, P, D, E]))
        assert not intersectible(RPath([D, P, F, B, F, P]), RPath([E, D, P, D, E]))
        assert not intersectible(RPath([E]), RPath([E, D, P, D, E]))

    def test_co_intersectible(self):
        # Example 1, Figure 5, in Lee and Honavar 2015 UAI workshop
        Ij, Ik, B, E1, E2, E3 = es = [E_Class(e, ()) for e in ["Ij", "Ik", "B", "E1", "E2", "E3"]]
        R1, R2, R3, R4, R5 = rs = [R_Class("R1", (), {B: Cardinality.one, E1: Cardinality.one}),
                                   R_Class("R2", (), {E1: Cardinality.one, E3: Cardinality.one}),
                                   R_Class("R3", (), {E1: Cardinality.one, E2: Cardinality.one}),
                                   R_Class("R4", (), {E2: Cardinality.one, E3: Cardinality.one, Ik: Cardinality.one}),
                                   R_Class("R5", (), {Ik: Cardinality.one, Ij: Cardinality.one})]
        Q = RPath([B, R1, E1, R2, E3, R4, Ik, R5, Ij])
        R = RPath([Ij, R5, Ik, R4, E3, R2, E1, R3, E2, R4, Ik])
        P = RPath([B, R1, E1, R3, E2, R4, Ik])
        P_prime = RPath([B, R1, E1, R2, E3, R4, Ik])

        assert P in set(extend(Q, R))
        assert intersectible(P, P_prime)
        assert P_prime == Q[:len(P_prime) - 1]
        assert not co_intersectible(Q, R, P, P_prime)

        # Twist with "many" cardinality
        Ij, Ik, B, E1, E2, E3 = es = [E_Class(e, ()) for e in ["Ij", "Ik", "B", "E1", "E2", "E3"]]
        R1, R2, R3, R4, R5 = rs = [R_Class("R1", (), {B: Cardinality.one, E1: Cardinality.one}),
                                   R_Class("R2", (), {E1: Cardinality.one, E3: Cardinality.one}),
                                   R_Class("R3", (), {E1: Cardinality.one, E2: Cardinality.one}),
                                   R_Class("R4", (), {E2: Cardinality.many, E3: Cardinality.many, Ik: Cardinality.one}),
                                   R_Class("R5", (), {Ik: Cardinality.one, Ij: Cardinality.one})]
        Q = RPath([B, R1, E1, R2, E3, R4, Ik, R5, Ij])
        R = RPath([Ij, R5, Ik, R4, E3, R2, E1, R3, E2, R4, Ik])
        P = RPath([B, R1, E1, R3, E2, R4, Ik])
        P_prime = RPath([B, R1, E1, R2, E3, R4, Ik])

        assert P in set(extend(Q, R))
        assert intersectible(P, P_prime)
        assert P_prime == Q[:len(P_prime) - 1]
        assert co_intersectible(Q, R, P, P_prime)

    def test_UnvisitedQueue(self):
        q = UnvisitedQueue()
        q.puts([1, 2, 3, 4, 1, 2, 3])
        assert len(q) == 4
        while q:
            q.pop()
        assert len(q) == 0
        q.puts([1, 2, 3, 4, 1, 2, 3])
        assert len(q) == 0

    def test_d_separated(self):
        g = PDAG()
        g.add_path([1, 2, 3])
        g.add_path([5, 4, 3])
        g.add_path([3, 6, 7, 8])
        g = g.as_networkx_dag()
        assert d_separated(g, 1, 5)
        assert not d_separated(g, 1, 2)
        assert not d_separated(g, 1, 3)
        assert not d_separated(g, 2, 3)
        assert not d_separated(g, 5, 3)
        assert not d_separated(g, 3, 1)
        assert d_separated(g, 1, 5)
        assert d_separated(g, 1, 4)
        assert d_separated(g, 2, 4)
        assert not d_separated(g, 2, 4, {3})
        assert not d_separated(g, 1, 5, {3})
        assert not d_separated(g, 1, 4, {3})
        assert not d_separated(g, 2, 4, {6})
        assert not d_separated(g, 1, 5, {7})
        assert not d_separated(g, 1, 4, {8})
        assert not d_separated(g, 1, 8)
        assert not d_separated(g, 5, 8)
        assert d_separated(g, 1, 8, {3})
        assert d_separated(g, 1, 8, {3, 5})
        assert d_separated(g, 1, 8, {3, 6})
        assert not d_separated(g, 1, 3, {8})
        assert not d_separated(g, 1, 3, {8, 6})


class TestAGG(unittest.TestCase):
    def test_company(self):
        rcm = company_rcm()
        agg2h = AbstractGroundGraph(rcm, 2 * rcm.max_hop)
        assert agg2h.agg.number_of_nodes() == 158
        assert agg2h.agg.number_of_edges() == 444

    def test_company_fig_4_4_maier(self):
        rcm = company_rcm()
        agg2h = AbstractGroundGraph(rcm, 2 * rcm.max_hop)
        # Fig 4.4 in Maier's Thesis
        E, P, B, D, F = EPBDF()
        rvs = [RVar([E, D, P, F, B, F, P], "Success"),
               RVar(E, "Competence"),
               RVar([E, D, P], "Success"),
               RVar([E, D, P, F, B], "Revenue"),
               RVar([E, D, P, D, E], "Competence"),
               RVar([E, D, P, D, E, D, P], "Success")]

        iv = frozenset((rvs[-1], rvs[0]))
        vs = set(rvs) | {iv}
        sub = agg2h.agg.subgraph(list(vs))
        assert len(sub.edges()) == 7
        assert {(rvs[0], rvs[3]),
                (rvs[1], rvs[2]),
                (rvs[2], rvs[3]),
                (rvs[4], rvs[2]),
                (rvs[4], iv),
                (rvs[4], rvs[5]),
                (iv, rvs[3])} == set(sub.edges())


class TestRpCD(unittest.TestCase):
    def test_sound_rules(self):
        g = PDAG()
        g.add_undirected_path([1, 2, 3, 4, 5])
        sound_rules(g)
        assert not g.oriented()
        assert len(g.unoriented()) == 4

        g.orient(1, 2)
        nc = {SymTriple(1, 2, 3)}
        sound_rules(g, nc)
        assert g.oriented() == {(1, 2), (2, 3)}

    def test_together(self):
        for _ in range(100):
            g = PDAG()
            # vs = np.arange(np.random.randint(20) + 1)
            # np.random.shuffle(vs)
            vs = np.random.permutation(np.random.randint(20) + 1)
            for x, y in combinations(vs, 2):
                if np.random.rand() < 0.2:
                    g.add_edge(x, y)

            # unshielded colliders
            nc = set()
            for y in vs:
                for x, z in combinations(g.pa(y), 2):
                    if not g.is_adj(x, z):
                        nc.add(SymTriple(x, y, z))

            flag = bool(np.random.randint(2))
            if flag:
                sound_rules(g, nc, bool(np.random.randint(2)))
            else:
                completes(g, nc)
            current = g.oriented()
            if not flag:
                sound_rules(g, nc, bool(np.random.randint(2)))
            else:
                completes(g, nc)
            post = g.oriented()
            assert current == post

    def test_completes(self):
        for _ in range(100):
            g = PDAG()
            # vs = np.arange(np.random.randint(20) + 1)
            # np.random.shuffle(vs)
            vs = np.random.permutation(np.random.randint(20) + 1)
            for x, y in combinations(vs, 2):
                if np.random.rand() < 0.2:
                    g.add_edge(x, y)

            # unshielded colliders
            nc = set()
            for y in vs:
                for x, z in combinations(g.pa(y), 2):
                    if not g.is_adj(x, z):
                        nc.add(SymTriple(x, y, z))

            completes(g, nc)
            current = g.oriented()
            sound_rules(g, nc, bool(np.random.randint(2)))
            post = g.oriented()
            assert current == post


def inner__(s):
    np.random.seed(s)

    schema = generate_schema()
    rcm = generate_rcm(schema, np.random.randint(1, 100), np.random.randint(1, 20), np.random.randint(0, 20))
    for d1 in rcm.directed_dependencies:
        for d2 in rcm.directed_dependencies:
            for PyVx in (d1, reversed(d1)):
                for QzVy in (d2, reversed(d2)):
                    if PyVx.cause.attr == QzVy.effect.attr:
                        for CUT in canonical_unshielded_triples(rcm, PyVx, QzVy, single=False,
                                                                with_anchors=False):
                            pass
                        for CUT, JJ in canonical_unshielded_triples(rcm, PyVx, QzVy, single=False,
                                                                    with_anchors=True):
                            pass
                        for CUT in canonical_unshielded_triples(rcm, PyVx, QzVy, single=True,
                                                                with_anchors=False):
                            pass
                        for CUT, JJ in canonical_unshielded_triples(rcm, PyVx, QzVy, single=True,
                                                                    with_anchors=True):
                            pass


class TestCUT(unittest.TestCase):
    def test_anchors_to_skeleton(self):
        pass

    def test_one_cut(self):
        pass

    def test_restore_anchors(self):
        pass

    @unittest.skip('time consuming')
    def test_possible_cases(self):
        n = 100
        seeds = [np.random.randint(np.iinfo(np.int32).max) for _ in range(n)]
        Parallel(-1)(delayed(inner__)(s) for s in seeds)


if __name__ == '__main__':
    unittest.main()
