import collections

from some_pkg.relational_domain import SkItem, RSchema
from some_pkg.relational_model import RVar, RPath, RCM
import networkx as nx
import typing

from collections import deque
# [((item, attribute), direction), ... ]


# TODO under development
class Structure:
    def __init__(self, schema: RSchema, base_item: SkItem, given):
        self.schema = schema
        self.perspective = base_item
        # Pseudo Skeleton where an relationship instance may not have all its entities.
        self.sk = nx.Graph()
        self.sk.add_node(self.perspective)
        self.paths = []


    def copy(self):
        pass

    def added_from(self, rpath, start, cause_effect_direction):
        pass

    def added_between(self, rpath, start: SkItem, end: SkItem, cause_effect_direction):
        assert rpath.base == start.item_class
        assert rpath.terminal == end.item_class
        if start == end and rpath.is_canonical:
            return {self, }
        pass

    @property
    def dpath(self):
        return None


class RCIResult:
    def __init__(self):
        pass


class RCIReasoner:
    def __init__(self, schema: RSchema, model: RCM):
        self.schema = schema
        self.model = model
        self.cdg = self.model.class_dependency_graph

    def is_ci(self, x: RVar, y: RVar, zs=None) -> RCIResult:
        if zs is None:
            zs = frozenset()
        elif isinstance(zs, RVar):
            zs = {zs, }
        assert len({z.base for z in zs}) == 1 and next(iter(zs)).base == x.base
        assert x.base == y.base

        # start with an empty structure
        base_item_class = x.base
        base_item = SkItem(0, base_item_class)
        structures = deque([Structure(self.schema, base_item, zs)])
        # add an edge (add a dependence)
        while structures:
            structure = structures.popleft()
            (item, attr), direction, given = structure.dpath.current
            item_class = item.item_class

            for rdep in self.model.directed_dependencies:
                if (rdep.effect.base, rdep.effect.attr) == (item_class, attr):
                    # P.Y --> Vx
                    if direction == '>':
                        if given:   # collider
                            structure.added_from()
                        else:       # collider + given
                            structure.added_between()
                elif (rdep.cause.terminal, rdep.cause.attr) == (item_class, attr):
                    # Q.X --> Vy
                    pass

        return None


if __name__ == '__main__':
    pass
