# Practical Learning Algorithm for RCM
import collections
import logging
from itertools import product, count, combinations
from warnings import warn

from some_pkg.domain import RSchema, RSkeleton
from some_pkg.model import PRCM, RVar, RPath, RDep, UndirectedRDep, canonical_rvars


def _groupby(xs, keyfunc):
    from itertools import groupby
    return {k: list(g) for k, g in groupby(sorted(xs, key=keyfunc), key=keyfunc)}.items()


def enumerate_rpaths(schema: RSchema, h_max: int):
    # canonical relational paths
    rpaths = collections.deque(RPath(item_class) for item_class in schema.entities | schema.relationships)
    while rpaths:
        rpath = rpaths.popleft()
        yield rpath
        if rpath.hop_len < h_max:
            nexts = schema.relateds(rpath.terminal)
            for appended_rpath in filter(lambda x: x is not None, (rpath.appended_or_none(n) for n in nexts)):
                rpaths.append(appended_rpath)


def enumerate_rdeps(schema: RSchema, h_max: int):
    assert 0 <= h_max

    for rpath in enumerate_rpaths(schema, h_max):
        for cause_attr, effect_attr in product(rpath.terminal.attrs, rpath.base.attrs):
            if cause_attr != effect_attr:
                cause = RVar(rpath, cause_attr)
                effect = RVar(RPath(rpath.base), effect_attr)
                yield RDep(cause, effect)


class PracticalLearner:
    def __init__(self, schema: RSchema, h_max: int, skeleton: RSkeleton, ci_tester):
        assert 0 <= h_max
        if 7 <= h_max:
            warn('high h_max {}'.format(h_max))

        self.schema = schema
        self.h_max = h_max
        self.skeleton = skeleton
        self.ci_tester = ci_tester

        # learning structure
        self.prcm = PRCM(self.schema)
        pass

    # do something with ci_test
    def adjacency_ci_test(self, cause, effect, conds, size):
        for cond in combinations(conds, size):
            is_ci, ci_result = self.ci_tester.test(cause, effect, cond)
            if is_ci:
                return True
        return False

    def orientation_ci_test(self):
        pass

    def phase_I(self) -> set:
        logging.info('phase I: started.')
        prcm, schema, ci_tester = self.prcm, self.schema, self.ci_tester

        # Initialize an undirected RCM
        udeps_to_be_tested = set(UndirectedRDep(dep) for dep in enumerate_rdeps(self.schema, self.h_max))
        prcm.add(udeps_to_be_tested)

        for d in count():
            to_remove = set()
            for udep in udeps_to_be_tested:
                for cause, effect in udep:
                    if self.adjacency_ci_test(cause, effect, prcm.ne(effect), d):
                        to_remove.add(udep)
                        break

            prcm.remove(to_remove)
            udeps_to_be_tested -= to_remove
            udeps_to_be_tested -= set(filter(lambda rvar: len(prcm.ne(rvar)) <= d, canonical_rvars(schema)))
            if not udeps_to_be_tested:
                break

        logging.info('phase I: finished.')
        return set(prcm.undirected_dependencies)

    def phase_II(self):
        # pcdg = self.prcm.class_dependency_graph
        # pgg = GroundGraph(self.prcm, self.skeleton)
        #
        # # find all unshielded triples
        # item_attr_uts = list(pgg.unshielded_triples())
        #
        # def attrfy(symtriple_of_item_attribute):
        #     (_, x), (_, y), (_, z) = symtriple_of_item_attribute
        #     return SymTriple(x, y, z)
        #
        # for (x, y, z), item_attr_triples in _groupby(item_attr_uts, attrfy):
        #     reordered = list()
        #     for t in item_attr_triples:
        #         (_, a), (_, b), (_, c) = t
        #         if (x, y, z) == (a, b, c):
        #             reordered.append(t)
        #         elif (x, y, z) == (c, b, a):
        #             reordered.append(t.dual)
        #         else:
        #             raise AssertionError('huh?')
        #
        #     self.orientation_ci_test(reordered)

        pass

    def phase_III(self):
        pass
