import functools

from pyrcds.domain import R_Class, E_Class, A_Class, Cardinality, RSchema, RSkeleton, SkItem
from pyrcds.model import RDep, PRCM
from pyrcds.model import RVar


@functools.lru_cache(1)
def EPBDF():
    E = E_Class('Employee', ('Salary', 'Competence'))
    P = E_Class('Product', (A_Class('Success'),))
    B = E_Class('BizUnit', (A_Class('Revenue'), A_Class('Budget')))
    D = R_Class('Develops', tuple(), {E: Cardinality.many, P: Cardinality.many})
    F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})

    return E, P, B, D, F


@functools.lru_cache(1)
def company_deps():
    E, P, B, D, F = EPBDF()
    deps = (RDep(RVar(E, 'Competence'), RVar(E, 'Salary')),
            RDep(RVar([E, D, P, F, B], 'Budget'), RVar(E, 'Salary')),
            RDep(RVar([P, D, E], 'Competence'), RVar(P, 'Success')),
            RDep(RVar([B, F, P], 'Success'), RVar(B, 'Revenue')),
            RDep(RVar(B, 'Revenue'), RVar(B, 'Budget')))
    return deps


@functools.lru_cache(1)
def company_schema():
    E, P, B, D, F = EPBDF()
    return RSchema({E, P, B}, {D, F})


@functools.lru_cache(1)
def company_skeleton():
    E, P, B, D, F = EPBDF()
    entities = ['Paul', 'Roger', 'Quinn', 'Sally', 'Thomas',
                'Case', 'Adapter', 'Laptop', 'Tablet', 'Smartphone',
                'Accessories', 'Devices']

    entity_types = {'Paul': E, 'Roger': E, 'Quinn': E, 'Sally': E, 'Thomas': E,
                    'Case': P, 'Adapter': P, 'Laptop': P, 'Tablet': P, 'Smartphone': P,
                    'Accessories': B, 'Devices': B}
    skeleton = RSkeleton(company_schema, True)
    p, r, q, s, t, c, a, l, ta, sm, ac, d = ents = tuple([SkItem(e, entity_types[e]) for e in entities])
    skeleton.add_entities(*ents)
    for emp, prods in ((p, {c, }), (q, {c, a, l}), (s, {l, ta}), (t, {sm, ta}), (r, {l, })):
        for prod in prods:
            skeleton.add_relationship(SkItem(emp.name + '-' + prod.name, D), {emp, prod})
    for biz, prods in ((ac, {c, a}), (d, {l, ta, sm})):
        for prod in prods:
            skeleton.add_relationship(SkItem(biz.name + '-' + prod.name, F), {biz, prod})

    return skeleton


@functools.lru_cache(1)
def company_rcm():
    return PRCM(company_schema(), company_deps())
