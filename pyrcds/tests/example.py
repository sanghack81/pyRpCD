from pyrcds.domain import *
from pyrcds.rcds import *

if __name__ == '__main__':
    # Please check Marc Maier's Thesis for the example here
    # Relational Schema
    E = E_Class('Employee', ('Salary', 'Competence'))
    P = E_Class('Product', 'Success')
    B = E_Class('BizUnit', ('Revenue', 'Budget'))
    D = R_Class('Develops', tuple(), {E: Cardinality.many, P: Cardinality.many})
    F = R_Class('Funds', tuple(), {P: Cardinality.one, B: Cardinality.many})

    schema = RSchema({E, P, B}, {D, F})

    # Relational Causal Model
    deps = (RDep(RVar(E, 'Competence'), RVar(E, 'Salary')),
            RDep(RVar([E, D, P, F, B], 'Budget'), RVar(E, 'Salary')),
            RDep(RVar([P, D, E], 'Competence'), RVar(P, 'Success')),
            RDep(RVar([B, F, P], 'Success'), RVar(B, 'Revenue')),
            RDep(RVar(B, 'Revenue'), RVar(B, 'Budget')))

    rcm = RCM(schema, deps)

    # Relational Skeleton
    skeleton = RSkeleton(schema, True)
    entity_names = ['Paul', 'Roger', 'Quinn', 'Sally', 'Thomas',
                    'Case', 'Adapter', 'Laptop', 'Tablet', 'Smartphone',
                    'Accessories', 'Devices']
    entity_types = {'Paul': E, 'Roger': E, 'Quinn': E, 'Sally': E, 'Thomas': E,
                    'Case': P, 'Adapter': P, 'Laptop': P, 'Tablet': P, 'Smartphone': P,
                    'Accessories': B, 'Devices': B}
    p, r, q, s, t, c, a, l, ta, sm, ac, d = ents = tuple([SkItem(e, entity_types[e]) for e in entity_names])
    skeleton.add_entities(*ents)
    for emp, prods in ((p, {c, }), (q, {c, a, l}), (s, {l, ta}), (t, {sm, ta}), (r, {l, })):
        for prod in prods:
            skeleton.add_relationship(SkItem(emp.name + '-' + prod.name, D), {emp, prod})
    for biz, prods in ((ac, {c, a}), (d, {l, ta, sm})):
        for prod in prods:
            skeleton.add_relationship(SkItem(biz.name + '-' + prod.name, F), {biz, prod})

    # Test for Markov Equivalence
    cprcm = markov_equivalence(rcm)
    for ud in cprcm.undirected_dependencies:
        print(ud)
    for d in cprcm.directed_dependencies:
        print(d)
    print()

    # Test for RpCD with Abstract Ground Graphs as a CI tester (possibly inaccurate)
    agg = AbstractGroundGraph(rcm, 2 * rcm.max_hop)
    rpcd = RpCD(schema, rcm.max_hop, agg)
    rpcd.phase_I()
    rpcd.phase_II()

    for ud in rpcd.prcm.undirected_dependencies:
        print(ud)
    for d in rpcd.prcm.directed_dependencies:
        print(d)

    assert rpcd.prcm == cprcm
