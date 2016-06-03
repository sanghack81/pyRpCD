import logging
import typing
import warnings
from collections import deque, defaultdict
from itertools import takewhile, count, combinations, chain

import networkx as nx

from pyrcds.domain import RSkeleton, RSchema, SkItem, R_Class, E_Class
from pyrcds.graphs import PDAG
from pyrcds.model import RDep, PRCM, llrsp, RVar, eqint, RPath, RCM, UndirectedRDep, canonical_rvars, SymTriple
from pyrcds.utils import group_by, safe_iter


def enumerate_rpaths(schema, hop, base_item_class=None):
    assert 0 <= hop
    Ps = deque()
    if base_item_class is not None:
        Ps.append(RPath(base_item_class))
    else:
        Ps.extend((RPath(ic) for ic in schema.item_classes))

    while Ps:
        P = Ps.pop()
        yield P
        if P.hop_len < hop:
            Ps.extend(filter(lambda x: x is not None, (P.appended_or_none(i) for i in schema.relateds(P.terminal))))


def enumerate_rvars(schema: RSchema, hop):
    for base_item_class in schema.item_classes:
        for P in enumerate_rpaths(schema, hop, base_item_class):
            for attr in P.terminal.attrs:
                yield RVar(P, attr)


class interner(dict):
    def __missing__(self, key):
        self[key] = key
        return key


def enumerate_rdeps(schema: RSchema, hop):
    c = interner()
    for base_item_class in schema.item_classes:
        if not base_item_class.attrs:
            continue
        for P in enumerate_rpaths(schema, hop, base_item_class):
            for cause_attr in P.terminal.attrs:
                for effect_attr in base_item_class.attrs:
                    if effect_attr != cause_attr:
                        yield RDep(c[RVar(P, cause_attr)], c[RVar(base_item_class, effect_attr)])


def extend(P: RPath, Q: RPath):
    assert P.terminal == Q.base
    m, n = len(P), len(Q)
    for pivot in takewhile(lambda piv: P[m - 1 - piv] == Q[piv], range(min(m, n))):
        if P[:m - 1 - pivot].joinable(Q[pivot:]):  # double?
            yield P[:m - 1 - pivot] ** Q[pivot:]


# See Lee and Honavar 2015
def intersectible(P: RPath, Q: RPath):
    if P == Q:
        raise AssertionError('{} == {}'.format(P, Q))
    return P.base == Q.base and P.terminal == Q.terminal and llrsp(P, Q) + llrsp(reversed(P), reversed(Q)) <= min(
        len(P), len(Q))


# See Lee and Honavar 2015
def co_intersectible(Q: RPath, R: RPath, P: RPath, P_prime: RPath):
    check = Q.terminal == R.base and \
            Q.base == P.base == P_prime.base and \
            R.terminal == P.terminal == P_prime.terminal and \
            intersectible(P, P_prime)

    if not check:
        raise AssertionError('not a valid arguments: {}'.format([Q, R, P, P_prime]))

    ll = 1 + (len(Q) + len(R) - 1 - len(P)) // 2
    Qm, Rp = Q[:len(Q) - ll], R[ll - 1:]
    l1, l2 = llrsp(Q, P_prime), llrsp(reversed(R), reversed(P_prime))
    return (l1 < len(Qm) or l2 < len(Rp)) and l1 + l2 <= len(P_prime)


class UnvisitedQueue:
    def __init__(self, iterable=()):
        self.visited = set(iterable)
        self.queue = deque(self.visited)

    def put(self, x):
        if x not in self.visited:
            self.visited.add(x)
            self.queue.append(x)

    def puts(self, xs):
        for x in xs:
            self.put(x)

    def __len__(self):
        return len(self.queue)

    def pop(self):
        return self.queue.popleft()

    def __bool__(self):
        return bool(self.queue)


def d_separated(dag: nx.DiGraph, x, y, zs=frozenset()):
    assert x != y
    assert x not in zs and y not in zs

    qq = UnvisitedQueue(((x, '>'), (x, '<')))
    while qq:
        node, direction = qq.pop()
        if direction == '>':
            if node not in zs:
                qq.puts((ch, '>') for ch in dag.successors_iter(node))
            else:
                qq.puts((pa, '<') for pa in dag.predecessors_iter(node))

        else:  # '<'
            if node not in zs:
                qq.puts((ch, '>') for ch in dag.successors_iter(node))
                qq.puts((pa, '<') for pa in dag.predecessors_iter(node))

        if {(y, '>'), (y, '<')} & qq.visited:
            return False

    return True


class AbstractGroundGraph:
    def __init__(self, rcm: RCM, h: int, n_jobs=1):
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        c1 = interner()  # memory-saver, takes time...
        c2 = interner()  # memory-saver, takes time...
        #
        self.RVs = set(c1[rv] for rv in enumerate_rvars(rcm.schema, h))
        #
        self.RVEs = set()
        self.IVs = set()
        self.IVEs = set()
        # IVs
        self.combined = defaultdict(set)
        for _, rvs in group_by(self.RVs, lambda rv: (rv.rpath.base, rv.attr)):
            for rv1, rv2 in combinations(rvs, 2):
                if intersectible(rv1.rpath, rv2.rpath):
                    self.IVs.add(c2[frozenset((rv1, rv2))])
                    self.combined[rv1.rpath].add(rv2.rpath)
                    self.combined[rv2.rpath].add(rv1.rpath)

                    if not (len(self.IVs) % 1000):
                        self.logger.info('creating {} of IVs.'.format(len(self.IVs)))
        # RVEs and IVEs
        for Y, Qys in group_by(self.RVs, lambda rv: rv.attr):
            for RxVy in filter(lambda d: d.effect.attr == Y, rcm.directed_dependencies):
                for Qy in Qys:
                    Q, (R, X) = Qy.rpath, RxVy.cause
                    for P in filter(lambda p: p.hop_len <= h, extend(Q, R)):
                        Px = c1[RVar(P, X)]
                        self.RVEs.add((Px, Qy))  # P.X --> Q.Y

                        if not (len(self.RVEs) % 1000):
                            self.logger.info('creating {} of RVEs.'.format(len(self.RVEs)))

                        # Q, R, P, P_prime
                        for P_prime in self.combined[P]:
                            if co_intersectible(Q, R, P, P_prime):
                                iv = c2[frozenset((Px, c1[RVar(P_prime, X)]))]
                                self.IVEs.add((iv, Qy))

                                if not (len(self.IVEs) % 1000):
                                    self.logger.info('creating {} of IVEs.'.format(len(self.IVEs)))
                        # P, reversed(R), Q, Q_prime
                        for Q_prime in self.combined[Q]:
                            if co_intersectible(P, reversed(R), Q, Q_prime):
                                iv = c2[frozenset((Qy, c1[RVar(Q_prime, Y)]))]
                                self.IVEs.add((Px, iv))

                                if not (len(self.IVEs) % 1000):
                                    self.logger.info('creating {} of IVEs.'.format(len(self.IVEs)))

        self.RVs = frozenset(self.RVs)
        self.RVEs = frozenset(self.RVEs)
        self.IVs = frozenset(self.IVs)
        self.IVEs = frozenset(self.IVEs)
        c1, c2 = None, None
        self.agg = nx.DiGraph()
        self.agg.add_nodes_from(self.RVs)
        self.agg.add_nodes_from(self.IVs)
        self.agg.add_edges_from(self.RVEs)
        self.agg.add_edges_from(self.IVEs)

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()):
        assert x != y
        assert x not in zs and y not in zs
        assert len({x.base, y.base} | {z.base for z in zs}) == 1
        assert y.is_canonical

        x_bar = {x} | self.combined[x]
        y_bar = {y} | self.combined[y]
        zs_bar = set(chain(*[{z} | self.combined[z] for z in zs]))

        x_bar -= zs_bar
        y_bar -= zs_bar

        if x_bar & y_bar:
            return False
        # all disjoint
        for x_ in x_bar:
            for y_ in y_bar:
                if not d_separated(self.agg, x_, y_, zs_bar):
                    return False
        return True

    @property
    def is_p_value_available(self):
        return False


class AbstractRCD:
    def __init__(self, schema, h_max, ci_tester):
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

        self.schema = schema
        self.h_max = h_max
        self.ci_tester = ci_tester

        self.sepset = defaultdict(lambda: None)
        self.prcm = PRCM(schema)

    def ci_test(self, cause: RVar, effect: RVar, conds: typing.Set[RVar], size: int):
        assert 0 <= size and effect.is_canonical

        for cond in combinations(conds, size):
            self.logger.debug('ci testing: {} _||_ {} | {}'.format(cause, effect, cond))
            ci_result = self.ci_tester.ci_test(cause, effect, cond)
            if ci_result.ci:
                self.logger.info('{} _||_ {} | {}'.format(cause, effect, cond))
                self.sepset[frozenset((cause, effect))] = cond
                return True
        return False

    def phase_I(self):
        """Find adjacencies of the underlying RCM"""
        self.logger.info('phase I: started.')
        prcm, schema, ci_tester = self.prcm, self.schema, self.ci_tester

        # Initialize an undirected RCM
        udeps_to_be_tested = set(UndirectedRDep(dep) for dep in enumerate_rdeps(self.schema, self.h_max))
        prcm.add(udeps_to_be_tested)

        for d in count():
            self.logger.info('phase I: checking depth: {}'.format(d))
            to_remove = set()
            for udep in udeps_to_be_tested:  # TODO parallelize
                for dep in udep:
                    self.logger.info('phase I: checking: {}'.format(dep))
                    cause, effect = dep
                    if self.ci_test(cause, effect, prcm.ne(effect), d):
                        to_remove.add(udep)
                        break

            # post clean up
            prcm.remove(to_remove)
            udeps_to_be_tested -= to_remove
            udeps_to_be_tested -= set(filter(lambda rvar: len(prcm.ne(rvar)) <= d, canonical_rvars(schema)))
            if not udeps_to_be_tested:
                break

        self.logger.info('phase I: finished.')
        return self.prcm

    def find_sepset(self, Vx, Rz):
        """Find a separating set only from the neighbors of canonical relational variable (i.e., Vx)"""
        assert Vx.is_canonical
        pair_key = frozenset((Vx, Rz))
        if self.sepset[pair_key] is not None:
            return self.sepset[pair_key]

        candidates = list(self.prcm.adj(Vx) - {Rz})
        for size in range(len(candidates) + 1):
            if self.ci_test(Rz, Vx, candidates, size):
                return self.sepset[pair_key]
        return None

    def phase_II(self):
        raise NotImplementedError()


def sound_rules(g: PDAG, non_colliders=(), purge=True):
    """Orient edges in the given PDAG where non-colliders information may not be completed."""
    while True:
        mark = len(g.oriented())
        for non_collider in tuple(non_colliders):
            x, y, z = non_collider
            # R1 X-->Y--Z (shielded, and unshielded)
            if g.is_oriented_as(x, y) and g.is_unoriented(y, z):
                g.orient(y, z)
            # R1' Z-->Y--X (shielded, and unshielded)
            if g.is_oriented_as(z, y) and g.is_unoriented(y, x):
                g.orient(y, x)

            # R3 (do not check x--z)
            for w in g.ch(x) & g.adj(y) & g.ch(z):
                g.orient(y, w)

            # R4   z-->w-->x
            if g.pa(x) & g.adj(y) & g.ch(z):
                g.orient(y, x)
            # R4'   x-->w-->z
            if g.pa(z) & g.adj(y) & g.ch(x):
                g.orient(y, z)

            if {x, z} <= g.ne(y):
                if z in g.ch(x):
                    g.orient(y, z)
                if x in g.ch(z):
                    g.orient(y, x)

        # R2
        for y in g:
            # x -> y ->z and x -- z
            for x in g.pa(y):
                for z in g.ch(y) & g.ne(x):
                    g.orient(x, z)

        # TODO non-colliders making an undirected cycle

        if purge:
            for non_collider in list(non_colliders):
                x, y, z = non_collider
                if (not (g.ne(y) & {x, z})) or ({x, z} & g.ch(y)):
                    non_colliders.discard(non_collider)

        if len(g.oriented()) == mark:
            break


def completes(g: PDAG, non_colliders):
    """Maximally orients edges in the given PDAG with (shielded or unshielded) non-collider constraints"""
    U = set(chain(*[[(x, y), (y, x)] for x, y in g.unoriented()]))

    # filter out directions, which violates non-collider constraints.
    # Nothing will be changed if sound rules (R1) are applied to g before 'completes' is called.
    for x, y in safe_iter(U):
        # x-->y
        if any(SymTriple(x, y, z) in non_colliders for z in g.pa(y)):
            g.orient(y, x)
            U -= {(x, y), (y, x)}

    for x, y in safe_iter(U):
        h = g.copy()
        h.orient(x, y)
        if ext(h, non_colliders):
            U -= g.oriented()
        else:
            g.orient(y, x)
            U -= {(x, y), (y, x)}


def ext(g: PDAG, NC):
    """Extensibility where non-colliders can be either shielded or unshielded"""
    h = g.copy()
    while h:
        for y in h:
            if not h.ch(y) and all(SymTriple(x, y, z) not in NC for x, z in combinations(h.adj(y), 2)):
                for x in h.ne(y):
                    g.orient(x, y)
                break
        else:
            return False
    return True


class RpCD(AbstractRCD):
    def __init__(self, schema, h_max, ci_tester):
        super().__init__(schema, h_max, ci_tester)

    def enumerate_CUTs(self):
        """Enumerate CUTs whose attribute classes are distinct"""
        by_cause_attr = defaultdict(set)
        by_effect_attr = defaultdict(set)
        for d in self.prcm.directed_dependencies:
            by_cause_attr[d.cause.attr].add(d)
            by_effect_attr[d.effect.attr].add(d)

        done = set()
        for Y in self.schema.attrs:
            for QzVy in by_effect_attr[Y]:
                for PyVx in by_cause_attr[Y]:
                    X, Z = PyVx.effect.attr, QzVy.cause.attr
                    # distinct only
                    if (X, Y, Z) not in done:
                        cut = canonical_unshielded_triples(self.prcm, PyVx, QzVy)
                        if cut is not None:
                            yield cut
                            done.add((X, Y, Z))  # ordered triple

    def phase_II(self, background_knowledge=()):
        """Orient undirected dependencies"""
        self.logger.info('phase II: started.')
        pcdg = self.prcm.class_dependency_graph

        if background_knowledge:
            for edge in background_knowledge:
                pcdg.orient(*edge)
            sound_rules(pcdg, set())

        NC = set()
        for Vx, PPy, Rz in self.enumerate_CUTs():
            X, Y, Z = Vx.attr, next(iter(PPy)).attr, Rz.attr

            # inactive non-collider / fully-oriented / already in non-colliders
            if ({X, Z} & pcdg.ch(Y)) or (not ({X, Z} & pcdg.ne(Y))) or (SymTriple(X, Y, Z) in NC):
                continue
            #
            sepset = self.find_sepset(Vx, Rz)
            if sepset is not None:  # can be checked with dual-RUT
                if not (PPy & sepset):
                    pcdg.orient(X, Y)
                    pcdg.orient(Z, Y)
                else:
                    if X == Z:  # RBO
                        pcdg.orient(Y, X)
                    else:
                        NC.add(SymTriple(X, Y, Z))
            sound_rules(pcdg, NC)

        completes(pcdg, NC)

        for x, y in pcdg.oriented():
            self.prcm.orient_with(x, y)

        self.logger.info('phase II: finished.')
        return self.prcm


def joinable(p, *args):
    for q in args:
        p = p ** q
        if p is None:
            return False
    return True


def restore_anchors(P, Q, a_r, b_r, a_s, b_s, a_t=None, b_t=None):
    """Given characteristic anchors, construct a fully specified a set of anchors."""
    last_P, first_Q = len(P) - 1, 0  # for readability

    # (...|P|-1, 0...)
    # (a_r..., ...b_r) (...a_r, ...b_r)
    # (a_s..., b_s...) (a_s..., ...b_s)
    J = {(last_P - i, first_Q + i) for i in range(llrsp(P[::-1], Q))} | \
        {(a_r + i, b_r - i) for i in range(llrsp(P[a_r:], Q[:b_r:-1]))} | \
        {(a_r - i, b_r - i) for i in range(llrsp(P[:a_r:-1], Q[:b_r:-1]))} | \
        {(a_s + i, b_s + i) for i in range(llrsp(P[a_s:], Q[b_s:]))} | \
        {(a_s + i, b_s - i) for i in range(llrsp(P[a_s:], Q[:b_s:-1]))}

    if a_t is not None and b_t is not None:
        # (a_t..., ...b_t) (...a_t, ...b_t)
        J |= {(a_t + i, b_t - i) for i in range(llrsp(P[a_t:], Q[:b_t:-1]))} | \
             {(a_t - i, b_t - i) for i in range(llrsp(P[:a_t:-1], Q[:b_t:-1]))}

    return J


def anchors_to_skeleton(schema: RSchema, P: RPath, Q: RPath, J):
    """Given anchors, construct a relational skeleton, which admits the anchors"""
    temp_g = nx.Graph()

    # Both
    pp = [None] * len(P)
    qq = [None] * len(Q)
    for a, b in J:
        pp[a] = qq[b] = SkItem('p' + str(a) + 'q' + str(b), P[a])
        temp_g.add_node(pp[a])

    # P only
    aa, bb = zip(*J)
    for a in set(range(len(P))) - set(aa):
        pp[a] = SkItem('p' + str(a), P[a])
        temp_g.add_node(pp[a])
    # Q only
    for b in set(range(len(Q))) - set(bb):
        qq[b] = SkItem('q' + str(b), Q[b])
        temp_g.add_node(qq[b])

    temp_g.add_edges_from(list(zip(pp, pp[1:])))
    temp_g.add_edges_from(list(zip(qq, qq[1:])))

    all_auxs = set()
    cc = count()
    for v in list(temp_g.nodes()):
        if isinstance(v.item_class, R_Class):
            missing = set(v.item_class.entities) - {ne.item_class for ne in temp_g.neighbors(v)}
            auxs = [SkItem('aux' + str(next(cc)), ic) for ic in missing]
            all_auxs |= auxs
            temp_g.add_nodes_from(auxs)
            for aux in auxs:
                temp_g.add_edge(v, aux)

    skeleton = RSkeleton(schema, True)
    entities = list(filter(lambda v: isinstance(v.item_class, E_Class), temp_g.nodes()))
    relationships = list(filter(lambda v: isinstance(v.item_class, R_Class), temp_g.nodes()))
    skeleton.add_entities(*entities)
    for r in relationships:
        skeleton.add_relationship(r, set(temp_g.neighbors(r)))

    return skeleton, all_auxs


# written for readability
# can be faster by employing view-class for RPath for slicing operator
def canonical_unshielded_triples(M: PRCM, PyVx: RDep, QzVy: RDep, single=True, with_anchors=False):
    """Returns a CUT or generate CUTs with/without anchors"""
    LL = llrsp

    Py, Vx = PyVx
    Qz, Vy = QzVy
    P, Y = Py
    Q, Z = Qz
    V, Y2 = Vy

    if Y != Y2:
        raise AssertionError("{} and {} do not share the common attribute class.".format(PyVx, QzVy))

    m, n = len(P), len(Q)
    l = LL(reversed(P), Q)
    a_x, b_x = m - l, l - 1

    # A set of candidate anchors
    J = set()
    for a in range(a_x + 1):  # 0 <= <= a_x
        for b in range(b_x, n):  # b_x <=  <= |Q|
            if P[a] == Q[b]:
                J.add((a, b))

    # the first characteristic anchor (a_r,b_r)
    for a_r, b_r in J:
        if not (LL(P[:a_r:-1], Q[b_r:]) == LL(P[a_r:], Q[b_r:]) == 1):
            continue
        if not joinable(P[:a_r], Q[b_r:]):
            continue
        RrZ = RVar(P[:a_r] ** Q[b_r:], Z)
        if RrZ in M.adj(Vx):
            continue

        l_alpha = LL(Q[b_x:b_r:-1], P[:a_r:-1])
        if l_alpha == 1:
            if eqint(P[a_r:a_x], Q[b_x:b_r:-1]):
                cut = (Vx, frozenset({Py, RVar(P[:a_r] ** Q[:b_r:-1], Y)}), RrZ)
                if single:
                    if with_anchors:
                        return cut, restore_anchors(P, Q, a_r, b_r, a_r, b_r)
                    else:
                        return cut
                else:
                    if with_anchors:
                        yield cut, restore_anchors(P, Q, a_r, b_r, a_r, b_r)
                    else:
                        yield cut

        elif l_alpha < b_r - b_x + 1 and a_r < a_x and b_x < b_r:
            a_y, b_y = a_r - l_alpha + 1, b_r - l_alpha + 1

            # the second characteristic anchor
            for a_s, b_s in J:
                if not (a_s <= a_y and b_x < b_s <= b_y):
                    continue
                if not joinable(P[:a_s], Q[b_s:]):
                    continue
                RsZ = RVar(P[:a_s] ** Q[b_s:], Z)
                if RsZ in M.adj(Vx):
                    continue

                PA, PB, QA, QB = P[:a_s:-1], P[a_s:a_y], Q[b_s:b_y], Q[b_x:b_s:-1]

                if LL(PA, QA) > 1 or LL(PA, QB) > 1:
                    continue

                l_beta = LL(PB, QB)
                if (not eqint(PB, QA)) or l_beta == min(len(PB), len(QB)):
                    continue

                a_z, b_z = a_s + l_beta - 1, b_s - l_beta + 1
                # the third characteristic anchor
                for a_t, b_t in J:
                    if not (a_r < a_t <= a_x and b_x <= b_t < b_z):
                        continue
                    if not joinable(P[:a_s], Q[b_t:b_s:-1], P[a_r:a_t:-1], Q[b_r:]):
                        continue
                    RtZ = RVar(P[:a_s] ** Q[b_t:b_s:-1] ** P[a_r:a_t:-1] ** Q[b_r:], Z)
                    if RtZ in M.adj(Vx):
                        continue

                    PC, PD, QC, QD = P[a_r:a_t:-1], P[a_t:a_x], Q[b_t:b_z], Q[b_x:b_t:-1]

                    if LL(PC, QC) > 1 or LL(PD, QC) > 1:
                        continue

                    l_gamma = LL(PC, QD)
                    assert 1 <= l_gamma
                    if l_gamma == 1 and eqint(PD, QD) or 1 < l_gamma < min(len(PC),
                                                                           len(QD)) and a_t < a_x and b_x < b_t:
                        a_w, b_w = a_t - l_gamma + 1, b_t - l_gamma + 1

                        PP = {P,
                              P[:a_w] ** Q[:b_w:-1],
                              P[:a_s] ** Q[:b_s:-1],
                              P[:a_s] ** Q[b_t:b_s:-1] ** P[a_t:],
                              P[:a_s] ** Q[b_s:b_r] ** P[a_r:],
                              P[:a_s] ** Q[b_s:b_r] ** P[a_r:a_w] ** Q[:b_w:-1]}

                        PP_Y = frozenset({RVar(PP_i, Y) for PP_i in PP})

                        if single:
                            if with_anchors:
                                return (Vx, PP_Y, RrZ), restore_anchors(P, Q, a_r, b_r, a_s, b_s, a_t, b_t)
                            else:
                                return Vx, PP_Y, RrZ
                        else:
                            if with_anchors:
                                JJ = restore_anchors(P, Q, a_r, b_r, a_s, b_s, a_t, b_t)
                                for R in {RrZ, RsZ, RtZ}:
                                    yield (Vx, PP_Y, R), JJ
                            else:
                                for R in {RrZ, RsZ, RtZ}:
                                    yield Vx, PP_Y, R
