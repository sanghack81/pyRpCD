import warnings
from collections import defaultdict
from enum import Enum
from functools import total_ordering
from itertools import chain

import networkx as nx


class Cardinality(Enum):
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
    def __init__(self, name: str):
        assert isinstance(name, str)
        assert bool(name)
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, SchemaElement) and self.name == other.name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name


class I_Class(SchemaElement):
    def __init__(self, name, attrs):
        assert attrs is not None
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
    def __init__(self, name, attrs):
        super().__init__(name, attrs)

    def removed(self, attrs: set):
        return E_Class(self.name, self.attrs - attrs)

    def __repr__(self):
        return self.name + "(" + repr(self.attrs) + ")"


class R_Class(I_Class):
    def __init__(self, name, attrs, cards: dict):
        super().__init__(name, attrs)
        self.__cards = cards.copy()

    # E in R
    def __contains__(self, item):
        return item in self.__cards

    def __getitem__(self, item):
        return self.__cards[item]

    def is_many(self, entity):
        return self.__cards[entity] == Cardinality.many

    @property
    def entities(self):
        return self.__cards.keys()

    def removed(self, to_remove: set):
        removed_attrs = self.attrs - to_remove
        removed_cards = {e: self.__cards[e] for e in self.entities - to_remove}
        if len(removed_cards) < 2:
            warnings.warn('relationship class with less than 2 entity classes: {}'.format(self.name))
        return R_Class(self.name, removed_attrs, removed_cards)

    def __repr__(self):
        return self.name + "(" + repr(self.attrs) + ", " + repr(
            {e.name: ('many' if self.is_many(e) else 'one') for e in self.__cards}) + ")"


class A_Class(SchemaElement):
    def __init__(self, name):
        super().__init__(name)

    def __repr__(self):
        return 'A_Class(' + repr(self.name) + ')'


class RSchema:
    # TODO just item_classes, separate in the code.
    def __init__(self, entities, relationships):
        assert all(isinstance(e, E_Class) for e in entities)
        assert all(isinstance(r, R_Class) for r in relationships)
        assert _is_unique((*_names(entities), *_names(relationships),
                           *[attr.name for item in (set(entities) | set(relationships)) for attr in item.attrs]))

        self.entities = frozenset(entities)
        self.relationships = frozenset(relationships)
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

    def item_class_of(self, attr):
        return self.attr2item_class[attr]

    @property
    def item_classes(self):
        return self.entities | self.relationships

    def __contains__(self, item):
        return item in self.entities or \
               item in self.relationships or \
               item in self.attrs

    def __str__(self):
        return "RSchema(" + ', '.join(e.name for e in (self.entities | self.relationships)) + ")"

    def __repr__(self):
        return "RSchema(" + repr(self.entities) + ", " + repr(self.relationships) + ")"

    def relateds(self, item_class: I_Class):
        assert isinstance(item_class, I_Class)
        return self.__i2i[item_class]

    def removed(self, to_remove):
        pass

    def as_networkx_ug(self):
        g = nx.Graph()
        g.add_nodes_from(self.entities)
        g.add_nodes_from(self.relationships)
        for r in self.relationships:
            g.add_edges_from([(e, r) for e in r.entities])
        return g


class SkItem:
    def __init__(self, name, item_class: I_Class, values: dict = None):
        assert hash(name)
        assert isinstance(item_class, I_Class)
        self.name = name
        self.item_class = item_class
        self.__values = values.copy() if values is not None else dict()

    def __eq__(self, other):
        if self is other:
            return True
        else:
            # lazy-check for different instance of the same name
            assert self.name != other.name
            return False

    def __hash__(self):
        return hash(self.name)

    def __getitem__(self, item: A_Class):
        return self.__values[item]

    def __setitem__(self, item: A_Class, value):
        self.__values[item] = value

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class RSkeleton:
    def __init__(self, strict=False):
        self.__G = nx.Graph()
        self.__nodes_by_type = defaultdict(set)
        self.__strict = strict

    def __setitem__(self, key, value):
        item, attr = key
        assert isinstance(item, SkItem) and isinstance(attr, A_Class)
        item[attr] = value

    def __getitem__(self, key):
        item, attr = key
        return item[attr]

    def add_entities(self, *args):
        for x in args:
            self.add_entity(x)

    def add_entity(self, item: SkItem):
        assert isinstance(item.item_class, E_Class)
        assert item not in self.__G
        self.__nodes_by_type[item.item_class].add(item)
        self.__G.add_node(item)

    def add_relationship(self, rel: SkItem, entities):
        assert isinstance(rel.item_class, R_Class)
        assert all(isinstance(e.item_class, E_Class) for e in entities)
        assert rel not in self.__G
        assert all(e in self.__G for e in entities)
        for e in entities:
            if not rel.item_class.is_many(e.item_class):
                assert len(self.neighbors(e, rel.item_class)) == 0

        if self.__strict:
            set(rel.item_class.entities) == {e.item_class for e in entities}

        self.__nodes_by_type[rel.item_class].add(rel)
        self.__G.add_node(rel)
        self.__G.add_edges_from((rel, e) for e in entities)

    def items(self, filter_type: I_Class = None):
        if filter_type is not None:
            return frozenset(self.__nodes_by_type[filter_type])
        return frozenset(self.__G)

    def neighbors(self, x, filter_type: I_Class = None):
        if filter_type is None:
            return frozenset(self.__G[x])
        else:
            return frozenset(filter(lambda y: y.item_class == filter_type, self.__G[x]))

    def __str__(self):
        return str(self.__G)

    def as_networkx_ug(self):
        return self.__G.copy()
