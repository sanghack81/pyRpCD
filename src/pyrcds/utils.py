import itertools

import networkx as nx
from numpy.random import choice, randint, random_sample, shuffle, randn

from pyrcds.domain import RSchema, RSkeleton, E_Class, A_Class, R_Class, SkItem, Cardinality, I_Class
from pyrcds.model import RPath, RCM, RVar, RDep, ParamRCM, terminal_set


#
class between_sampler:
    def __init__(self, min_inclusive, max_inclusive):
        assert min_inclusive <= max_inclusive
        self.m = min_inclusive
        self.M = max_inclusive

    def sample(self, size=None):
        if size is None:
            return randint(self.m, self.M + 1)
        else:
            return randint(self.m, self.M + 1, size=size).tolist()


#
class cardinality_sampler:
    def __init__(self, p_many=0.5):
        assert 0 <= p_many <= 1.0
        self.p_many = p_many

    def sample(self):
        if random_sample() <= self.p_many:
            return Cardinality.many
        else:
            return Cardinality.one


#
class normal_sampler:
    def __init__(self, mu=0.0, sd=1.0):
        self.mu = mu
        self.sd = sd

    def sample(self):
        return self.sd * randn() + self.mu


# sample() = sample(1)
# sample(n)
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


# Erdos-Renyi
def generate_skeleton(schema: RSchema, n_entities: dict = None, n_relationships: dict = None) -> RSkeleton:
    if n_entities is None:
        n_entities = {E: randint(300, 500) for E in schema.entities}
    if n_relationships is None:
        n_relationships = {R: randint(300, 500) for R in schema.relationships}

    skeleton = RSkeleton(schema, strict=True)
    counter = itertools.count(1)

    # adjust number of entities if more relationships are 'requesting'
    # if E in R with cardinality one, |\sigma(R)| <= |\sigma(E)|.
    for R in schema.relationships:
        for E in R.entities:
            if not R.is_many(E) and n_relationships[R] > n_entities[E]:
                n_entities[E] = n_relationships[R]

    entities = {E: [SkItem("e" + str(next(counter)), E) for _ in range(n_entities[E])] for E in schema.entities}
    for vs in entities.values():
        for v in vs:
            skeleton.add_entity(v)

    for R in schema.relationships:
        selected = {E: choice(entities[E], n_relationships[R], replace=R.is_many(E)).tolist()
                    for E in R.entities}
        for i in range(n_relationships[R]):
            ents = [selected[E][i] for E in R.entities]
            skeleton.add_relationship(SkItem("r" + str(next(counter)), R), ents)

    return skeleton


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

    return RPath(rpath_inner)


def generate_rcm(schema: RSchema, num_dependencies, max_degree, max_hop):
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
            cause = RVar(RPath(rpath), cause_attr)
            candidate = RDep(cause, effect)
            if candidate not in rcm.directed_dependencies:
                rcm.add(candidate)
                failed_count = 0
            else:
                failed_count += 1

    if len(rcm.directed_dependencies) > num_dependencies:
        return RCM(schema, choice(list(rcm.directed_dependencies), num_dependencies).tolist())
    return rcm


def average_agg(default=0.0):
    def func(items):
        if len(items) > 0:
            return sum(items) / len(items)
        else:
            return default

    return func


def max_agg(default=0.0):
    def func(items):
        if len(items) > 0:
            return max(items)
        else:
            return default

    return func


def linear_gaussian(parameters: dict, aggregator, error):
    """
    a linear model with an additive Gaussian noise
    :param parameters: parameters for linear model. parameter = parameters[cause_rvar]
    :param aggregator: a function that maps multiple values to a single value.
    :param error: additive noise distribution, err = error.sample()
    :return: a function that can be used in a parametrized RCM
    """

    def func(values, cause_item_attrs):
        value = 0
        for rvar in parameters:
            item_attr_values = [values[item_attr] for item_attr in cause_item_attrs[rvar]]
            value += parameters[rvar] * aggregator(item_attr_values)
        return value + error.sample()

    return func


# randomly generated parameters
# linear additive Gaussian
def linear_gaussians_rcm(rcm: RCM):
    functions = dict()
    effects = {RVar(RPath(rcm.schema.item_class_of(attr)), attr) for attr in rcm.schema.attrs}

    for e in effects:
        parameters = {cause: 1.0 + 0.1 * abs(randn()) for cause in rcm.pa(e)}
        functions[e] = linear_gaussian(parameters, average_agg(), normal_sampler(0, 0.1))

    return ParamRCM(rcm.schema, rcm.directed_dependencies, functions)


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
