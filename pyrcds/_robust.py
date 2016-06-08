# Practical Learning Algorithm for RCM
import itertools

from model import GroundGraph, SymTriple
from pyrcds.rcds import AbstractRCD
from utils import group_by


class PracticalLearner(AbstractRCD):
    def __init__(self, schema, h_max, ci_tester, verbose=False):
        super().__init__(schema, h_max, ci_tester, verbose)

    def phase_II(self, background_knowledge=None, truth=None, verbose=False, **kwargs):
        pcdg = self.prcm.class_dependency_graph
        pgg = GroundGraph(self.prcm, self.skeleton)

        if background_knowledge:
            pcdg.orients(background_knowledge)

        # find all unshielded triples
        item_attr_uts = list(pgg.unshielded_triples())

        def attrfy(symtriple_of_item_attribute):
            (_, x), (_, y), (_, z) = symtriple_of_item_attribute
            return SymTriple(x, y, z) if x < z else SymTriple(z, y, x)

        # TODO address conflict?
        # TODO parallelize
        self.ori_results = {(x, y, z): self.collider_ci_test((x, y, z), item_attr_triples)
                            for (x, y, z), item_attr_triples in group_by(item_attr_uts, attrfy)}

        singletons = {}
        for (x, y, z), result in self.ori_results.items():
            if result.is_collider:
                ppcdg = pcdg.copy()
                ppcdg.orients(((x, y), (z, y)))
                singletons[frozenset({(x, y, z)})] = (ppcdg, set(), result.score)
            elif result.is_non_collider:
                if x == z:
                    ppcdg = pcdg.copy()
                    ppcdg.orient(y, x)
                    singletons[frozenset({(x, y, z)})] = (ppcdg, set(), result.score)
                else:
                    singletons[frozenset({(x, y, z)})] = (pcdg.copy(), {(x, y, z)}, result.score)

        maximals = {1: singletons}
        for size in itertools.count(2):
            for m1, m2 in itertools.combinations(maximals[size - 1].keys(), 2):
                if len(m1 & m2) == size - 2 and self.checkcheck(m1, m2, maximals[size - 1]):
                    maximals[size].update({})

            if not maximals[size]:
                break

        maximals.values()
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
