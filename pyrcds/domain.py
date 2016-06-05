import functools
import itertools
import typing
from collections import defaultdict
from enum import Enum
from functools import total_ordering
from itertools import chain

import networkx as nx
from numpy.random.mtrand import random_sample, choice, randint

from pyrcds.utils import between_sampler


class Cardinality(Enum):
    """Define constants for how many relationships an entity can participate in"""
    one = 1
    many = 2

    def __str__(self):
        return type(self).__name__ + '.' + self.name

    def __repr__(self):
        return type(self).__name__ + '.' + self.name


def _names(ys) -> set:
    return {y.name for y in ys}


def _is_unique(ys) -> bool:
    return len(set(ys)) == len(ys)


@total_ordering
class SchemaElement:
    """An abstract class for item classes and attribute classes"""

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError('A name must be a non-empty string')
        # assert isinstance(name, str)
        # assert bool(name)
        self.name = name
        self.__h = hash(self.name)

    def __hash__(self):
        return self.__h

    def __eq__(self, other):
        return isinstance(other, SchemaElement) and self.name == other.name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name


class I_Class(SchemaElement):
    """An item class of a relational schema"""

    def __init__(self, name, attrs=()):
        if isinstance(attrs, A_Class):
            attrs = {attrs, }
        elif isinstance(attrs, str):
            attrs = {A_Class(attrs), }
        attrs = {A_Class(a) if isinstance(a, str) else a for a in attrs}
        assert all(isinstance(a, A_Class) for a in attrs)
        assert name not in _names(attrs)

        super().__init__(name)
        self.attrs = frozenset(attrs)


class E_Class(I_Class):
    """An entity class of a relational schema"""

    def __init__(self, name, attrs=()):
        super().__init__(name, attrs)

    def __repr__(self):
        attr_part = ', '.join(str(a) for a in sorted(self.attrs)) if self.attrs else ''
        return self.name + "(" + attr_part + ")"


class R_Class(I_Class):
    """A relationship class of a relational schema"""

    def __init__(self, name, attrs, cards: dict):
        super().__init__(name, attrs)
        self.__cards = cards.copy()
        self.entities = frozenset(self.__cards.keys())

    # E in R
    def __contains__(self, item):
        """Whether an entity class participates in this relationship class"""
        return item in self.__cards

    def __getitem__(self, item):
        """Cardinality of a participating entity class"""
        return self.__cards[item]

    def is_many(self, entity):
        return self.__cards[entity] == Cardinality.many

    def __repr__(self):
        attr_part = ', '.join(str(a) for a in sorted(self.attrs)) if self.attrs else '()'
        card_part = '{' + (
            ', '.join(
                [str(e) + ': ' + ('many' if self.is_many(e) else 'one') for e in sorted(self.__cards.keys())])) + '}'
        return self.name + "(" + attr_part + ", " + card_part + ")"


class A_Class(SchemaElement):
    def __init__(self, name):
        super().__init__(name)

    def __repr__(self):
        return 'A_Class(' + repr(self.name) + ')'


# Immutable
class RSchema:
    def __init__(self, entities, relationships):
        assert all(isinstance(e, E_Class) for e in entities)
        assert all(isinstance(r, R_Class) for r in relationships)
        assert _is_unique((*_names(entities), *_names(relationships),
                           *[attr.name for item in (set(entities) | set(relationships)) for attr in item.attrs]))

        self.entities = frozenset(entities)
        self.relationships = frozenset(relationships)
        self.item_classes = self.entities | self.relationships
        self.attrs = frozenset(chain(*[i.attrs for i in chain(entities, relationships)]))

        __i2i = defaultdict(set)
        for r in relationships:
            __i2i[r] = r.entities
            for e in r.entities:
                __i2i[e].add(r)
        for e in self.entities - __i2i.keys():
            __i2i[e] = set()

        self.__i2i = {i: frozenset(__i2i[i]) for i in __i2i}
        self.attr2item_class = dict()
        for item_class in self.entities | self.relationships:
            for attr in item_class.attrs:
                self.attr2item_class[attr] = item_class

        self.elements = {e.name: e for e in self.entities | self.relationships | self.attrs}

    def __getitem__(self, name) -> SchemaElement:
        """Returns a schema element given its name"""
        return self.elements[name]

    def item_class_of(self, attr) -> I_Class:
        """Returns an item class which has the given attribute class"""
        return self.attr2item_class[attr]

    def __contains__(self, item):
        """Whether the given schema element is in this relational schema"""
        return item in self.entities or \
               item in self.relationships or \
               item in self.attrs

    def __str__(self):
        return "RSchema(" + ', '.join(e.name for e in sorted(self.entities | self.relationships)) + ")"

    def __repr__(self):
        return "RSchema(Entity classes: " + repr(sorted(self.entities)) + ", Relationship classes: " + repr(
            sorted(self.relationships)) + ")"

    def relateds(self, item_class: I_Class) -> frozenset:
        """Returns neighboring item classes"""
        return self.__i2i[item_class]

    def as_networkx_ug(self, with_attribute_classes=False) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(self.entities)
        g.add_nodes_from(self.relationships)
        for r in self.relationships:
            g.add_edges_from([(e, r) for e in r.entities])
        if with_attribute_classes:
            g.add_nodes_from(self.attrs)
            for attr in self.attrs:
                g.add_edge(self.item_class_of(attr), attr)
        return g


@functools.total_ordering
class SkItem:
    def __init__(self, name, item_class: I_Class, values: dict = None):
        self.name = name
        self.item_class = item_class
        self.__values = values.copy() if values is not None else dict()
        self.__h = hash(self.name)

    def __eq__(self, other):
        return isinstance(other, SkItem) and self.name == other.name

    def __hash__(self):
        return self.__h

    def __le__(self, other):
        return self.name <= other.name

    def __contains__(self, k):
        """Whether the value of given attribute is set"""
        return k in self.__values

    def __getitem__(self, item):
        """Get the value of the given attribute"""
        if item not in self.__values:
            return None
        return self.__values[item]

    def __setitem__(self, item, value):
        """Set the value of the given attribute"""
        self.__values[item] = value

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class RSkeleton:
    def __init__(self, schema: RSchema, strict=False):
        self.schema = schema
        self._G = nx.Graph()
        self._nodes_by_type = defaultdict(set)
        self.__strict = strict

    def __setitem__(self, key, value):
        item, attr = key
        assert isinstance(item, SkItem)
        item[attr] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            pass
        else:
            item, attr = key
            if attr not in item:
                return None
            return item[attr]

    def add_entities(self, *args):
        for x in args:
            self.add_entity(x)

    def add_entity(self, item: SkItem):
        assert isinstance(item.item_class, E_Class)
        assert item not in self._G
        self._nodes_by_type[item.item_class].add(item)
        self._G.add_node(item)

    def add_relationship(self, rel: SkItem, entities):
        assert isinstance(rel.item_class, R_Class)
        assert all(isinstance(e.item_class, E_Class) for e in entities)
        assert rel not in self._G
        assert all(e in self._G for e in entities)
        for e in entities:
            if not rel.item_class.is_many(e.item_class):
                assert len(self.neighbors(e, rel.item_class)) == 0

        if self.__strict:
            assert set(rel.item_class.entities) == {e.item_class for e in entities}

        self._nodes_by_type[rel.item_class].add(rel)
        self._G.add_node(rel)
        self._G.add_edges_from((rel, e) for e in entities)

    def items(self, filter_type: I_Class = None) -> typing.FrozenSet[SkItem]:
        """Returns items of the given type, if provided"""
        if filter_type is not None:
            return frozenset(self._nodes_by_type[filter_type])
        return frozenset(self._G)

    def neighbors(self, x, filter_type: I_Class = None):
        """Returns x's neighboring items of the given type, if provided"""
        if filter_type is None:
            return frozenset(self._G[x])
        else:
            return frozenset(filter(lambda y: y.item_class == filter_type, self._G[x]))

    def __str__(self):
        return str(self._G)

    def as_networkx_ug(self) -> nx.Graph:
        return self._G.copy()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class ImmutableRSkeleton(RSkeleton):
    def __init__(self, skeleton: RSkeleton):
        self.schema = skeleton.schema
        self._nodes = frozenset(skeleton._G.nodes_iter())
        self._nodes_of = defaultdict(frozenset)
        self._nodes_of.update({k: frozenset(vs) for k, vs in skeleton._nodes_by_type.items()})
        self._ne = defaultdict(frozenset)
        for v in self._nodes:
            neighbors = skeleton.neighbors(v)
            self._ne[(v, None)] = frozenset(neighbors)
            for k, g in itertools.groupby(sorted(neighbors, key=lambda x: x.item_class),
                                          key=lambda x: x.item_class):
                self._ne[(v, k)] = frozenset(g)
        self._G = nx.freeze(skeleton.as_networkx_ug())

    def __setitem__(self, key, value):
        raise AssertionError('not allowed to modify')

    def __getitem__(self, key):
        item, attr = key
        return item[attr]

    def add_entities(self, *args):
        raise AssertionError('not allowed to modify')

    def add_entity(self, item: SkItem):
        raise AssertionError('not allowed to modify')

    def add_relationship(self, rel: SkItem, entities):
        raise AssertionError('not allowed to modify')

    def items(self, filter_type: I_Class = None):
        if filter_type is not None:
            return self._nodes_of[filter_type]
        return self._nodes

    def neighbors(self, x, filter_type: I_Class = None):
        return self._ne[(x, filter_type)]

    def __str__(self):
        return str(self._G)

    def as_networkx_ug(self):
        return self._G.copy()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class cardinality_sampler:
    def __init__(self, p_many=0.5):
        assert 0 <= p_many <= 1.0
        self.p_many = p_many

    def sample(self):
        if random_sample() <= self.p_many:
            return Cardinality.many
        else:
            return Cardinality.one


def generate_schema(num_ent_classes_distr=between_sampler(2, 5),
                    num_rel_classes_distr=between_sampler(2, 5),
                    num_ent_classes_per_rel_class_distr=between_sampler(2, 3),
                    num_attr_classes_per_ent_class_distr=between_sampler(2, 4),
                    num_attr_classes_per_rel_class_distr=between_sampler(0, 0),
                    cardinality_distr=cardinality_sampler(0.5)  # Cardinality sampler
                    ):
    ent_classes = []
    rel_classes = []
    attr_count = itertools.count(1)

    num_ent_classes = num_ent_classes_distr.sample()
    if num_ent_classes < 2:
        num_rel_classes = 0
    else:
        num_rel_classes = num_rel_classes_distr.sample()

    for i in range(1, num_ent_classes + 1):
        n_attr = num_attr_classes_per_ent_class_distr.sample()
        attrs = (A_Class("A" + str(next(attr_count))) for _ in range(n_attr))
        ent_classes.append(E_Class("E" + str(i), attrs))
    assert len(ent_classes) == num_ent_classes

    for i in range(1, num_rel_classes + 1):
        n_e_r = num_ent_classes_per_rel_class_distr.sample()
        n_e_r = max(min(n_e_r, num_ent_classes), 2)
        cards = {ent_classes[i]: cardinality_distr.sample() for i in choice(num_ent_classes, n_e_r, replace=False)}
        n_attr = num_attr_classes_per_rel_class_distr.sample()
        attrs = (A_Class("A" + str(next(attr_count))) for _ in range(n_attr))
        rel_classes.append(R_Class("R" + str(i), attrs, cards))
    assert len(rel_classes) == num_rel_classes

    return RSchema(ent_classes, rel_classes)


def generate_skeleton(schema: RSchema, n_items=(300, 500), maximum_degrees=None) -> RSkeleton:
    # TODO implement maximum
    if isinstance(n_items, int):
        n_items = {ic: n_items for ic in schema.item_classes}
    elif isinstance(n_items, tuple):
        n_items = {ic: randint(*n_items) for ic in sorted(schema.item_classes)}
    if isinstance(maximum_degrees, int):
        maximum_degrees = {(r, e): maximum_degrees for r in schema.relationships for e in r.entities}

    skeleton = RSkeleton(schema, strict=True)
    counter = itertools.count(1)

    # adjust number of entities if more relationships are 'requesting'
    # if E in R with cardinality one, |\sigma(R)| <= |\sigma(E)|.
    for R in schema.relationships:
        for E in R.entities:
            if not R.is_many(E) and n_items[R] > n_items[E]:
                n_items[E] = n_items[R]

    entities = {E: [SkItem("e" + str(next(counter)), E) for _ in range(n_items[E])] for E in sorted(schema.entities)}
    for vs in entities.values():
        for v in vs:
            skeleton.add_entity(v)

    for R in sorted(schema.relationships):
        selected = {E: choice(entities[E], n_items[R], replace=R.is_many(E)).tolist()
                    for E in sorted(R.entities)}
        for i in range(n_items[R]):
            ents = [selected[E][i] for E in R.entities]
            skeleton.add_relationship(SkItem("r" + str(next(counter)), R), ents)

    return skeleton

#
# def generate_skeleton2(schema: RSchema, n_items=(300, 500), maximum_degrees=None, approximate_sizes=None) -> RSkeleton:
#     if isinstance(n_items, int):
#         n_items = {ic: n_items for ic in schema.item_classes}
#     elif isinstance(n_items, tuple):
#         n_items = {ic: randint(*n_items) for ic in schema.item_classes}
#     if isinstance(maximum_degrees, int):
#         maximum_degrees = {(r, e): maximum_degrees for r in schema.relationships for e in r.entities}
#
#     entities = defaultdict(list)
#     relationships = defaultdict(list)
#     degrees = defaultdict(lambda: 0)  # (r,e)
#
#     def entity_generator(E):
#         c = counter()
#         while True:
#             yield SkItem('e' + str(next(c)), E)
#
#     def rel_generator(R):
#         c = counter()
#         while True:
#             yield SkItem('r' + str(next(c)), R)
#
#     expected_size = 300
#     min_size = 100
#     max_size = 1000
#
#     num_rels = np.random.poisson(expected_size, len(schema.relationships))
#     num_rels[num_rels < min_size] = min_size
#     num_rels[num_rels > max_size] = max_size
#
#     skeleton = RSkeleton(schema, strict=True)
#     counter = itertools.count(1)
#
#     # adjust number of entities if more relationships are 'requesting'
#     # if E in R with cardinality one, |\sigma(R)| <= |\sigma(E)|.
#     for R in schema.relationships:
#         for E in R.entities:
#             if not R.is_many(E) and n_items[R] > n_items[E]:
#                 n_items[E] = n_items[R]
#
#     entities = {E: [SkItem("e" + str(next(counter)), E) for _ in range(n_items[E])] for E in schema.entities}
#     for vs in entities.values():
#         for v in vs:
#             skeleton.add_entity(v)
#
#     for R in schema.relationships:
#         selected = {E: choice(entities[E], n_items[R], replace=R.is_many(E)).tolist()
#                     for E in R.entities}
#         for i in range(n_items[R]):
#             ents = [selected[E][i] for E in R.entities]
#             skeleton.add_relationship(SkItem("r" + str(next(counter)), R), ents)
#
#     return skeleton
