from collections import defaultdict
from itertools import cycle, groupby

import networkx as nx
import numpy as np
from pandas import DataFrame

from some_pkg.graphs import PDAG
from some_pkg.relational_domain import E_Class, R_Class, I_Class, RSchema, A_Class, RSkeleton, SkItem


def is_valid_relational_path(path) -> bool:
    E, R = E_Class, R_Class
    assert path is not None and len(path) >= 1
    assert isinstance(path[0], E) or isinstance(path[0], R)

    # alternating sequence
    if isinstance(path[0], E):
        classes = cycle([E, R])
    else:
        classes = cycle([R, E])
    if not all(isinstance(item_class, cls) for item_class, cls in zip(path, classes)):
        return False

    # participation
    def is_related(x, y):
        return (x in y.entities) if isinstance(y, R) else (y in x.entities)

    if not all(is_related(x, y) for x, y in zip(path, path[1:])):
        return False

    # ERE, RER constraints
    zipped = zip(path[::2], path[1::2], path[2::2])
    shfited_zipped = zip(path[1::2], path[2::2], path[3::2])
    if isinstance(path[0], E):
        ere, rer = zipped, shfited_zipped
    else:
        ere, rer = shfited_zipped, zipped
    return all(e1 != e2 for e1, r, e2 in ere) and all(r1 != r2 or r1.is_many(e) for r1, e, r2 in rer)


class RPath:
    def __init__(self, item_classes):
        assert item_classes is not None
        if isinstance(item_classes, I_Class):
            item_classes = [item_classes, ]
        assert is_valid_relational_path(item_classes)
        self.__item_classes = tuple(item_classes)
        self.__h = hash(self.__item_classes)

    def __hash__(self):
        return self.__h

    def __iter__(self):
        return iter(self.__item_classes)

    def __eq__(self, other):
        return isinstance(other, RPath) and \
               self.__h == other.__h and \
               self.__item_classes == other.__item_classes

    def __getitem__(self, item):
        return self.__item_classes[item]

    @property
    def is_canonical(self):
        return len(self.__item_classes) == 1

    def subpath(self, start, end):
        assert 0 <= start < end <= len(self)
        return RPath(self.__item_classes[start:end])

    def __reversed__(self):
        return RPath(tuple(reversed(self.__item_classes)))

    @property
    def terminal(self):
        return self.__item_classes[-1]

    @property
    def base(self):
        return self.__item_classes[0]

    def __len__(self):
        return len(self.__item_classes)

    def joinable(self, rpath):
        if self.terminal != rpath.base:
            return False

        if len(self) > 1 and len(rpath) > 1:
            return is_valid_relational_path([self.__item_classes[-2], self.__item_classes[-1], rpath.__item_classes[1]])
        return True

    def join(self, rpath):
        assert self.terminal == rpath.base

        if len(self) > 1 and len(rpath) > 1:
            assert is_valid_relational_path([self.__item_classes[-2], self.__item_classes[-1], rpath.__item_classes[1]])

        return RPath(self.__item_classes + rpath.__item_classes[1:])

    def __str__(self):
        return '[' + (', '.join(str(i) for i in self.__item_classes)) + ']'


# Longest Length of Required Shared Path
def LLRSP(p1: RPath, p2: RPath) -> int:
    prev = None
    for i, (x, y) in enumerate(zip(p1, p2)):
        if x != y or (i > 0 and isinstance(x, R_Class) and x.is_many(prev)):
            return i
        prev = x

    return min(len(p1), len(p2))


# equal or intersectible
def eqint(p1: RPath, p2: RPath):
    if (p1.base, p1.terminal) != (p2.base, p2.terminal):
        return False
    return p1 == p2 or LLRSP(p1, p2) + LLRSP(reversed(p1), reversed(p2)) <= min(len(p1), len(p2))


class RVar:
    def __init__(self, rpath: RPath, attr: A_Class):
        if not isinstance(rpath, RPath):
            rpath = RPath(rpath)
        if isinstance(attr, str):
            attr = A_Class(attr)
        assert attr in rpath.terminal.attrs
        self.rpath = rpath
        self.attr = attr
        self.__h = hash(self.rpath) ^ hash(self.attr)

    @property
    def terminal(self):
        return self.rpath.terminal

    @property
    def base(self):
        return self.rpath.base

    @property
    def is_canonical(self):
        return self.rpath.is_canonical

    def __len__(self):
        return len(self.rpath)

    def __hash__(self):
        return self.__h

    def __eq__(self, other):
        return isinstance(other, RVar) and \
               self.__h == other.__h and \
               self.rpath == other.rpath and \
               self.attr == other.attr

    def __str__(self):
        return str(self.rpath) + '.' + str(self.attr)


class RDep:
    def __init__(self, cause: RVar, effect: RVar):
        assert effect.is_canonical
        assert cause.attr != effect.attr
        self.cause = cause
        self.effect = effect
        self.__h = hash(self.cause) ^ hash(self.effect)

    def __hash__(self):
        return self.__h

    def __eq__(self, other):
        return isinstance(other, RDep) and \
               self.cause == other.cause and \
               self.effect == other.effect

    def __len__(self):
        return len(self.cause.rpath)

    def __iter__(self):
        return iter([self.cause, self.effect])

    @property
    def hop_len(self) -> int:
        return len(self) - 1

    # reversed(P.X --> Vy) = ~P.Y --> Vx
    def __reversed__(self):
        new_cause = RVar(reversed(self.cause.rpath), self.effect.attr)
        new_effect = RVar(RPath(self.cause.terminal), self.cause.attr)
        return RDep(new_cause, new_effect)

    # dual(P.X --> Vy) = (Vx, ~P.Y)
    @property
    def dual(self):
        new_cause = RVar(reversed(self.cause.rpath), self.effect.attr)
        new_effect = RVar(RPath(self.cause.terminal), self.cause.attr)
        return new_effect, new_cause

    def attrfy(self):
        return self.cause.attr, self.effect.attr

    def __str__(self):
        return str(self.cause) + " -> " + str(self.effect)


# class SymTriple:
#     def __init__(self, left, middle, right):
#         assert left != right
#         self.left, self.middle, self.right = left, middle, right
#
#     def __hash__(self):
#         return (hash(self.left) + hash(self.right)) ^ hash(self.middle)
#
#     def __eq__(self, other):
#         return isinstance(other, SymTriple) and \
#                (self.left, self.right, self.middle) == (other.left, other.right, other.middle) or \
#                (self.right, self.left, self.middle) == (other.left, other.right, other.middle)
#
#     def sides(self):
#         return set(self.left, self.right)
#
#     def __str__(self):
#         return '<' + (', '.join((self.left, self.middle, self.right))) + '>'


# class AttrTriple(SymTriple):
#     def __init__(self, l, m, r):
#         super().__init__(l, m, r)


class UndirectedRDep:
    def __init__(self, rdep: RDep):
        assert isinstance(rdep, RDep)
        self.rdeps = frozenset({rdep, reversed(rdep)})

    def __eq__(self, other):
        return isinstance(other, UndirectedRDep) and self.rdeps == other.rdeps

    def __hash__(self):
        return hash(self.rdeps)

    def __iter__(self):
        return iter(self.rdeps)

    def hop_length(self):
        return next(iter(self.rdeps)).hop_len

    def __str__(self):
        c, e = next(iter(self))
        return str(c) + " -- " + str(e)


class PRCM:
    def __init__(self, schema, dependencies=None):
        if dependencies is None:
            dependencies = frozenset()
        self.schema = schema

        self.directed_dependencies = set()
        self.undirected_dependencies = set()

        self.parents = defaultdict(set)
        self.children = defaultdict(set)
        self.neighbors = defaultdict(set)

        self.add(dependencies)

    def pa(self, rvar: RVar):
        return self.parents[rvar]

    def ch(self, rvar: RVar):
        return self.children[rvar]

    def ne(self, rvar: RVar):
        return self.neighbors[rvar]

    def adj(self, rvar: RVar):
        return self.neighbors[rvar] | self.parents[rvar] | self.children[rvar]

    # -1 if there is no depedency
    @property
    def max_hop(self) -> int:
        a = max(len(v) for k, vs in self.parents.items() for v in vs) if self.parents else 0
        b = max(len(v) for k, vs in self.neighbors.items() for v in vs) if self.neighbors else 0
        return -1 + max(a, b)

    @property
    def class_dependency_graph(self):
        cdg = PDAG()
        cdg.add_edges((cause.attr, effect.attr) for effect, causes in self.parents.items() for cause in causes)
        cdg.add_undirected_edges((k.attr, v.attr) for k, vs in self.neighbors.items() for v in vs)
        return cdg

    def add(self, d):
        if isinstance(d, RDep):
            cause, effect = d  # Px --> Vy
            if d in self.directed_dependencies:
                return
            if UndirectedRDep(d) in self.undirected_dependencies:
                raise AssertionError('undirected dependency exists for {}'.format(d))
            if reversed(d) in self.directed_dependencies:
                raise AssertionError('opposite-directed dependency exists for {}'.format(d))
            self.parents[effect].add(cause)
            dual_cause, dual_effect = d.dual
            self.children[dual_cause].add(dual_effect)
            self.directed_dependencies.add(d)
        elif isinstance(d, UndirectedRDep):
            d1, d2 = d
            if d in self.undirected_dependencies:
                return
            if d1 in self.directed_dependencies:
                raise AssertionError('directed dependency {} exists'.format(d1))
            if d2 in self.directed_dependencies:
                raise AssertionError('directed dependency {} exists'.format(d2))
            self.neighbors[d1.effect].add(d1.cause)
            self.neighbors[d2.effect].add(d2.cause)
            self.undirected_dependencies.add(d)
        else:
            for x in d:  # delegate as far as it is iterable
                self.add(x)

    def remove(self, d):
        if isinstance(d, RDep) and d in self.directed_dependencies:
            self.parents[d.effect].remove(d.cause)
            self.children[d.dual.cause].discard(d.dual.effect)
            self.directed_dependencies.remove(d)

        elif isinstance(d, UndirectedRDep) and d in self.undirected_dependencies:
            d1, d2 = d
            self.neighbors[d1.effect].remove(d1.cause)
            self.neighbors[d2.effect].discard(d2.cause)
            self.undirected_dependencies.remove(d)

        else:
            for x in d:  # delegate as far as it is iterable
                self.remove(x)

    def orient_as(self, edge):
        if edge in self.directed_dependencies:
            return False
        assert isinstance(edge, RDep)
        cause, effect = edge
        assert cause in self.ne(effect)
        self.remove(UndirectedRDep(edge))
        self.add(edge)
        return True


class RCM(PRCM):
    def __init__(self, schema: RSchema, dependencies=None):
        super().__init__(schema, dependencies)

    def add(self, d):
        assert not isinstance(d, UndirectedRDep)
        super().add(d)


# function takes values and cause item attributes
# values[RVar]
# cause_item_attr[RVar] = tuples of an item and an attribute
class ParametrizedRCM(RCM):
    def __init__(self, schema: RSchema, dependencies, functions: dict):
        super().__init__(schema, dependencies)
        self.functions = functions


# class CanonicalUnshieldedTriple:
#     def __init__(self, l: RVar, ms, r: RVar):
#         assert l.is_canonical
#         self.left, self.middles, self.right = l, frozenset(ms), r
#
#     def __eq__(self, other):
#         return isinstance(other, CanonicalUnshieldedTriple) and \
#                (self.left, self.middles, self.right) == (other.left, other.middles, other.right)
#
#     def __hash__(self):
#         return hash((self.left, self.middles, self.right))
#
#     @property
#     def sides(self):
#         return self.left, self.right
#
#     def __str__(self):
#         pass
#
#     def attr_triple(self) -> AttrTriple:
#         return AttrTriple(self.left.attr, next(iter(self.middles)).attr, self.right.attr)


# TODO, speed up by employing a tree structure (memory efficient)
def terminal_set(skeleton: RSkeleton, rpath: RPath, base_item: SkItem):
    item_paths = [[base_item]]
    next_paths = []
    for item_class in rpath[1:]:
        for item_path in item_paths:
            next_items = skeleton.neighbors(item_path[-1], item_class) - set(item_path)
            next_paths += [item_path + [item, ] for item in next_items]
        item_paths = next_paths
        next_paths = []

    return {path[-1] for path in item_paths}


def flatten(skeleton: RSkeleton, rvars) -> DataFrame:
    """
    Create a data frame with rows representing base items and columns representing values of relational variables.
    Multiset will be represented as a list of tuples of an item and its value.
    :param skeleton: relational skeleton where values are stored.
    :param rvars: relational variables, which will be correspondent to columns.
    :return: a data frame with rows representing base items and columns representing values of relational variables.
    """
    rvars = list(rvars)
    assert len({rvar.base for rvar in rvars}) == 1
    base_class = rvars[0].base
    base_items = list(skeleton.items(base_class))

    data = np.empty([len(base_items), len(rvars)], dtype=object)
    for i, base_item in enumerate(base_items):
        for j, rvar in enumerate(rvars):
            terminal = terminal_set(skeleton, rvar.rpath, base_item)
            data[i, j] = [(item, item[rvar.attr]) for item in terminal]

    return DataFrame(data, base_items, rvars, dtype=object, )


class GroundGraph:
    def __init__(self, rcm: RCM, skeleton: RSkeleton):
        self.g = nx.DiGraph()

        def key_func(d):
            return d.effect.rpath.terminal

        for terminal, rdeps in groupby(sorted(rcm.directed_dependencies, key=key_func), key=key_func):
            for base_item in skeleton.items(terminal):
                for rdep in rdeps:  # same perspective
                    for dest_item in terminal_set(skeleton, rdep.cause.rpath, base_item):
                        self.g.add_edge((dest_item, rdep.cause.attr), (base_item, rdep.effect.attr))

    def __str__(self):
        return str(self.g)

    def as_networkx_dag(self):
        return self.g.copy()
