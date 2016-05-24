import functools
import typing
import warnings
from collections import defaultdict
from itertools import cycle, product, combinations

import networkx as nx
import numpy as np
from numpy.random.mtrand import choice, shuffle, randint, randn

from pyrcds.domain import E_Class, R_Class, I_Class, RSchema, A_Class, RSkeleton, SkItem
from pyrcds.graphs import PDAG
from pyrcds.utils import average_agg, normal_sampler, group_by, linear_gaussian


def is_valid_rpath(path) -> bool:
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
    def __init__(self, item_classes, backdoor=False):
        assert item_classes is not None
        if isinstance(item_classes, I_Class):
            item_classes = (item_classes,)
        assert backdoor or is_valid_rpath(item_classes)
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

    def __bool__(self):
        return True

    # As in the paper
    def __getitem__(self, item):
        if isinstance(item, int):
            return self.__item_classes[item]
        elif isinstance(item, slice):
            # TODO slice to tuple, lru cache, a few ...
            start = 0 if item.start is None else item.start
            stop = len(self) if item.stop is None else item.stop + 1
            assert 0 <= start < stop <= len(self)

            if item.step == -1:
                return RPath(tuple(reversed(self.__item_classes[start:stop])), True)
            else:
                return RPath(self.__item_classes[start:stop], True)

        else:
            raise AssertionError('unknown {}'.format(item))

    # path concatenation
    def __pow__(self, other):
        if self.joinable(other):
            return self.join(other, True)
        return None

    @property
    def is_canonical(self):
        return len(self.__item_classes) == 1

    def subpath(self, start, end):
        assert 0 <= start < end <= len(self)
        return RPath(self.__item_classes[start:end])

    def __reversed__(self):
        return RPath(tuple(reversed(self.__item_classes)), True)

    @property
    def hop_len(self):
        return len(self) - 1

    @property
    def terminal(self):
        return self.__item_classes[-1]

    @property
    def base(self):
        return self.__item_classes[0]

    def __len__(self):
        return len(self.__item_classes)

    # TODO test
    def appended_or_none(self, item_class: I_Class):
        if len(self) > 1:
            if is_valid_rpath(self.__item_classes[-2:] + (item_class,)):
                return RPath(self.__item_classes + (item_class,), True)
        else:
            if isinstance(item_class, R_Class):
                if self.terminal in item_class.entities:
                    return RPath(self.__item_classes + (item_class,), True)
            elif isinstance(self.terminal, R_Class):
                if item_class in self.terminal.entities:
                    return RPath(self.__item_classes + (item_class,), True)
        return None

    def joinable(self, rpath):
        if self.terminal != rpath.base:
            return False
        if len(self) > 1 and len(rpath) > 1:
            return is_valid_rpath([self.__item_classes[-2], self.__item_classes[-1], rpath.__item_classes[1]])
        return True

    def join(self, rpath, backdoor=False):
        if not backdoor:
            assert self.joinable(rpath)
        return RPath(self.__item_classes + rpath.__item_classes[1:], True)

    def __str__(self):
        return '[' + (', '.join(str(i) for i in self.__item_classes)) + ']'

    def __repr__(self):
        return str(self)


# TODO
class RPathView(RPath):
    def __init__(self, rpath, from_inc, to_inc):
        warnings.warn('not yet')
        self.inner_rpath = rpath
        self.from_inc = from_inc
        self.to_inc = to_inc

    def __hash__(self):
        pass

    def __iter__(self):
        return
        pass

    def __eq__(self, other):
        pass

    def __bool__(self):
        pass

    # As in the paper
    def __getitem__(self, item):
        pass

    # path concatenation
    def __pow__(self, other):
        pass

    @property
    def is_canonical(self):
        pass

    def subpath(self, start, end):
        pass

    def __reversed__(self):
        pass

    @property
    def hop_len(self):
        pass

    @property
    def terminal(self):
        pass

    @property
    def base(self):
        pass

    def __len__(self):
        pass

    # TODO test
    def appended_or_none(self, item_class: I_Class):
        pass

    def joinable(self, rpath):
        pass

    def join(self, rpath, backdoor=False):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass


# Longest Length of Required Shared Path
@functools.lru_cache(3)
def llrsp(p1: RPath, p2: RPath) -> int:
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
    return p1 == p2 or llrsp(p1, p2) + llrsp(reversed(p1), reversed(p2)) <= min(len(p1), len(p2))


# Immutable
class RVar:
    def __init__(self, rpath, attr: typing.Union[str, A_Class]):
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

    def __iter__(self):
        return iter((self.rpath, self.attr))

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

    def __repr__(self):
        return str(self)


def canonical_rvars(schema: RSchema) -> typing.Set[RVar]:
    """Returns all canonical relational variables given schema"""
    return set(RVar(RPath(item_class), attr)
               for item_class in schema.item_classes
               for attr in item_class.attrs)


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

    def __repr__(self):
        return str(self)


class SymTriple:
    def __init__(self, left, middle, right):
        self.left, self.middle, self.right = left, middle, right

    def __hash__(self):
        return (hash(self.left) + hash(self.right)) ^ hash(self.middle)

    def __eq__(self, other):
        return isinstance(other, SymTriple) and \
               (self.left, self.right, self.middle) == (other.left, other.right, other.middle) or \
               (self.right, self.left, self.middle) == (other.left, other.right, other.middle)

    def __iter__(self):
        return iter((self.left, self.middle, self.right))

    def sides(self):
        return set(self.left, self.right)

    @property
    def dual(self):
        return SymTriple(self.right, self.middle, self.left)

    def __str__(self):
        return '<' + (', '.join(str(t) for t in (self.left, self.middle, self.right))) + '>'


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

    @property
    def hop_len(self) -> int:
        return next(iter(self.rdeps)).hop_len

    def __str__(self):
        c, e = next(iter(self))
        return str(c) + " -- " + str(e)

    def attrfy(self):
        dep = next(iter(self.rdeps))
        return frozenset({dep.cause.attr, dep.effect.attr})


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
    def class_dependency_graph(self) -> PDAG:
        # TODO Not a view, create new one every time?
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
            dual_cause, dual_effect = d.dual
            self.children[dual_cause].discard(dual_effect)
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

    def orient_with(self, x, y):
        for udep in list(self.undirected_dependencies):
            if udep.attrfy() == frozenset({x, y}):
                for dep in udep:
                    if dep.attrfy() == (x, y):
                        self.orient_as(dep)


class RCM(PRCM):
    def __init__(self, schema: RSchema, dependencies=None):
        super().__init__(schema, dependencies)

    def add(self, d):
        assert not isinstance(d, UndirectedRDep)
        super().add(d)


# function takes values and cause item attributes
# values[RVar]
# cause_item_attr[RVar] = tuples of an item and an attribute
class ParamRCM(RCM):
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

    iterator = iter(rpath)
    next(iterator)
    for item_class in iterator:
        for item_path in item_paths:
            next_items = skeleton.neighbors(item_path[-1], item_class) - set(item_path)
            next_paths += [item_path + [item, ] for item in next_items]

        if not next_paths:
            return set()

        item_paths = next_paths
        next_paths = []

    assert all(len(path) == len(rpath) for path in item_paths)
    return {path[-1] for path in item_paths}


def flatten(skeleton: RSkeleton, rvars, with_base_items=False, value_only=False):
    rvars = list(rvars)
    assert len({rvar.base for rvar in rvars}) == 1
    base_class = rvars[0].base
    base_items = list(skeleton.items(base_class))

    data = np.empty([len(base_items), (1 if with_base_items else 0) + len(rvars)], dtype=object)
    if with_base_items:
        data[:, 0] = base_items

    for i, base_item in enumerate(base_items):
        for j, rvar in enumerate(rvars, start=1 if with_base_items else 0):
            terminal = terminal_set(skeleton, rvar.rpath, base_item)
            if value_only:
                data[i, j] = tuple(item[rvar.attr] for item in terminal)
            else:
                data[i, j] = tuple((item, item[rvar.attr]) for item in terminal)

    return data


class GroundGraph:
    def __init__(self, rcm: RCM, skeleton: RSkeleton):
        self.schema = skeleton.schema
        self.skeleton = skeleton
        self.rcm = rcm
        self.g = PDAG()

        def k_fun(d):
            return d.effect.rpath.base

        # TODO refactoring
        # n.b. this is only valid with path semantics
        # With bridge burning semantics, a partially-directed ground graph is not well-defined.
        as_rdeps = {dep1 for dep1, _ in rcm.undirected_dependencies}
        for base_item_class, rdeps in group_by(as_rdeps, k_fun):
            for base_item, rdep in product(skeleton.items(base_item_class), rdeps):
                for dest_item in terminal_set(skeleton, rdep.cause.rpath, base_item):
                    self.g.add_undirected_edge((dest_item, rdep.cause.attr), (base_item, rdep.effect.attr))

        for base_item_class, rdeps in group_by(rcm.directed_dependencies, k_fun):
            for base_item, rdep in product(skeleton.items(base_item_class), rdeps):
                for dest_item in terminal_set(skeleton, rdep.cause.rpath, base_item):
                    self.g.add_edge((dest_item, rdep.cause.attr), (base_item, rdep.effect.attr))

    def __str__(self):
        return str(self.g)

    def as_networkx_dag(self):
        return self.g.as_networkx_dag().copy()

    def unshielded_triples(self):
        uts = set()
        for middle in self.g:
            for left, right in combinations(self.g.adj(middle), 2):
                if not self.g.is_adj(left, right):
                    uts.add(SymTriple(left, middle, right))
        return uts


def generate_rpath(schema: RSchema, base: I_Class = None, length=None):
    assert length is None or 1 <= length
    if base is None:
        base = choice(list(schema.entities | schema.relationships))
    assert base in schema

    rpath_inner = [base, ]
    curr_item = base
    prev_item = None
    while len(rpath_inner) < length:
        next_items = set(schema.relateds(curr_item))
        if prev_item is not None:
            if isinstance(curr_item, R_Class) or not prev_item.is_many(curr_item):
                next_items.remove(prev_item)
        if not next_items:
            break
        next_item = choice(list(next_items))
        rpath_inner.append(next_item)

        prev_item = curr_item
        curr_item = next_item

    return RPath(rpath_inner, True)


def generate_rcm(schema: RSchema, num_dependencies=10, max_degree=5, max_hop=6):
    FAILED_LIMIT = len(schema.entities) + len(schema.relationships)
    # ordered attributes
    attr_order = list(schema.attrs)
    shuffle(attr_order)

    def causable(cause_attr_candidate):
        return attr_order.index(cause_attr_candidate) < attr_order.index(effect_attr)

    # schema may not be a single component
    rcm = RCM(schema)

    for effect_attr in attr_order:
        base_class = schema.item_class_of(effect_attr)
        effect = RVar(RPath(base_class), effect_attr)

        degree = randint(1, max_degree + 1)  # 1<= <= max_degree

        failed_count = 0
        while len(rcm.pa(effect)) < degree and failed_count < FAILED_LIMIT:
            rpath = generate_rpath(schema, base_class, randint(1, max_hop + 1 + 1))
            cause_attr_candidates = list(filter(causable, rpath.terminal.attrs - {effect_attr, }))
            if not cause_attr_candidates:
                failed_count += 1
                continue

            cause_attr = choice(cause_attr_candidates)
            cause = RVar(rpath, cause_attr)
            candidate = RDep(cause, effect)
            if candidate not in rcm.directed_dependencies:
                rcm.add(candidate)
                failed_count = 0
            else:
                failed_count += 1

    if len(rcm.directed_dependencies) > num_dependencies:
        return RCM(schema, choice(list(rcm.directed_dependencies), num_dependencies).tolist())
    return rcm


def _item_attributes(items, attr: A_Class):
    return {(item, attr) for item in items}


def generate_values_for_skeleton(rcm: ParamRCM, skeleton: RSkeleton):
    """
    Generate values for the given skeleton based on functions specified in the parametrized RCM.
    :param rcm: a parameterized RCM, where its functions are used to generate values on skeleton.
    :param skeleton: a skeleton where values will be assigned to its item-attributes
    """
    cdg = rcm.class_dependency_graph
    nx_cdg = cdg.as_networkx_dag()
    ordered_attributes = nx.topological_sort(nx_cdg)

    for attr in ordered_attributes:
        base_item_class = rcm.schema.item_class_of(attr)
        effect = RVar(RPath(base_item_class), attr)
        causes = rcm.pa(effect)

        for base_item in skeleton.items(base_item_class):
            cause_item_attrs = {cause: _item_attributes(terminal_set(skeleton, cause.rpath, base_item), cause.attr)
                                for cause in causes}

            v = rcm.functions[effect](skeleton, cause_item_attrs)
            skeleton[(base_item, attr)] = v


def linear_gaussians_rcm(rcm: RCM):
    functions = dict()
    effects = {RVar(RPath(rcm.schema.item_class_of(attr)), attr) for attr in rcm.schema.attrs}

    for e in effects:
        parameters = {cause: 1.0 + 0.1 * abs(randn()) for cause in rcm.pa(e)}
        functions[e] = linear_gaussian(parameters, average_agg(), normal_sampler(0, 0.1))

    return ParamRCM(rcm.schema, rcm.directed_dependencies, functions)
