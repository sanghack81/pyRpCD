# Practical Learning Algorithm for RCM
import collections
import logging
import typing
from itertools import count, combinations
from warnings import warn

from pyrcds.__rci import CITester
from pyrcds.domain import RSchema, RSkeleton
from pyrcds.model import PRCM, RVar, UndirectedRDep, canonical_rvars, RCM
from pyrcds.rcds import enumerate_rdeps


# def group_by(xs, keyfunc):
#     """Modified groupby from itertools with group objects as lists"""
#     from itertools import groupby
#     gb = groupby(sorted(xs, key=keyfunc), key=keyfunc)
#     return ((k, list(g)) for k, g in gb)  # generator


def sound_rules(pcdg_xyz):
    pass


# a set of compatible elements.
# a maximal set of compatible elements.


def checkcheck(m1, m2):
    pass


class PracticalLearner:
    def __init__(self, schema: RSchema, h_max: int, skeleton: RSkeleton, ci_tester: CITester):
        assert 0 <= h_max
        if 7 <= h_max:
            warn('high h_max {}'.format(h_max))

        self.schema = schema
        self.h_max = h_max
        self.skeleton = skeleton
        self.ci_tester = ci_tester

        self.sepset = collections.defaultdict(lambda: None)

        # intermediate structure
        self.prcm = PRCM(self.schema)
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def adjacency_ci_test(self, cause: RVar, effect: RVar, conds: typing.Set[RVar], size: int):
        assert 0 <= size and effect.is_canonical

        for cond in combinations(conds, size):
            self.logger.debug('ci testing: {} _||_ {} | {}'.format(cause, effect, cond))
            ci_result = self.ci_tester.ci_test(cause, effect, cond)
            if ci_result.ci:
                self.logger.info('{} _||_ {} | {}'.format(cause, effect, cond))
                self.sepset[(cause, effect)] = cond
                return True
        return False

    # def collider_ci_test(self):
    #     return False

    def phase_I(self, truth: RCM=False, verbose=False, **kwargs) -> set:
        self.logger.info('phase I: started.')
        prcm, schema, ci_tester = self.prcm, self.schema, self.ci_tester

        # Initialize an undirected RCM
        udeps_to_be_tested = set(UndirectedRDep(dep) for dep in enumerate_rdeps(self.schema, self.h_max))
        prcm.add(udeps_to_be_tested)

        for d in count():
            self.logger.info('phase I: checking depth: {}'.format(d))
            to_remove = set()
            for udep in udeps_to_be_tested:
                for dep in udep:
                    self.logger.info('phase I: checking: {}'.format(dep))
                    cause, effect = dep
                    if self.adjacency_ci_test(cause, effect, prcm.ne(effect), d):
                        if truth:
                            if cause in truth.adj(effect):
                                if verbose:
                                    print('False: {} _||_ {} | {}'.format(cause, effect, self.sepset[(cause, effect)]))
                            else:
                                if verbose:
                                    print('False: {} _||_ {} | {}'.format(cause, effect, self.sepset[(cause, effect)]))
                        to_remove.add(udep)
                        break
                else:
                    pass

            # post clean up
            prcm.remove(to_remove)
            udeps_to_be_tested -= to_remove
            udeps_to_be_tested -= set(filter(lambda rvar: len(prcm.ne(rvar)) <= d, canonical_rvars(schema)))
            if not udeps_to_be_tested:
                break

        self.logger.info('phase I: finished.')
        return set(prcm.undirected_dependencies)

    # def phase_II(self, background_knowledge=None, truth=None,verbose=False, **kwargs):
    #     pcdg = self.prcm.class_dependency_graph
    #     pgg = GroundGraph(self.prcm, self.skeleton)
    #
    #     if background_knowledge:
    #         pcdg.orients(background_knowledge)
    #
    #     # find all unshielded triples
    #     item_attr_uts = list(pgg.unshielded_triples())
    #
    #     def attrfy(symtriple_of_item_attribute):
    #         (_, x), (_, y), (_, z) = symtriple_of_item_attribute
    #         return SymTriple(x, y, z) if x < z else SymTriple(z, y, x)
    #
    #     # TODO address conflict?
    #     # TODO parallelize
    #     self.ori_results = {(x, y, z): self.collider_ci_test((x, y, z), item_attr_triples)
    #                         for (x, y, z), item_attr_triples in group_by(item_attr_uts, attrfy)}
    #
    #     singletons = {}
    #     for (x, y, z), result in self.ori_results.items():
    #         if result.is_collider:
    #             ppcdg = pcdg.copy()
    #             ppcdg.orients(((x, y), (z, y)))
    #             singletons[frozenset({(x, y, z)})] = (ppcdg, set(), result.score)
    #         elif result.is_non_collider:
    #             if x == z:
    #                 ppcdg = pcdg.copy()
    #                 ppcdg.orient(y, x)
    #                 singletons[frozenset({(x, y, z)})] = (ppcdg, set(), result.score)
    #             else:
    #                 singletons[frozenset({(x, y, z)})] = (pcdg.copy(), {(x, y, z)}, result.score)
    #
    #     maximals = {1: singletons}
    #     for size in itertools.count(2):
    #         for m1, m2 in combinations(maximals[size - 1].keys(), 2):
    #             if len(m1 & m2) == size - 2 and self.checkcheck(m1, m2, maximals[size - 1]):
    #                 maximals[size].update({})
    #
    #         if not maximals[size]:
    #             break
    #
    #     maximals.values()
    #
    # def checkcheck(self, m1, m2, reference):
    #     ppcdg1, nc1, score1 = reference[m1]
    #     ppcdg2, nc2, score2 = reference[m2]
    #     orients = ppcdg1.oriented() | ppcdg2.oriented()
    #     if any((y, x) in orients for x, y in orients):
    #         return False
    #     dag = nx.DiGraph()
    #     dag.add_edges_from(orients)
    #     if not nx.is_directed_acyclic_graph(dag):
    #         return False
    #         # sound rules
    #         # PDAG extensibility
