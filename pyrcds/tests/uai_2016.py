# To make things easier, each item class only has an attribute class

# Draw a schema

# Draw a skeleton

# Draw an RCM

# Draw a ground graph


# Pick two relational dependencies


# where they yield unshielded triples and shielded triples.


# Find out all CUTs
# show coverage
import json
from itertools import chain, count

from pyrcds.domain import generate_schema, generate_skeleton, RSkeleton, E_Class, RSchema
from pyrcds.model import generate_rcm, GroundGraph
from pyrcds.rcds import canonical_unshielded_triples, anchors_to_skeleton
from pyrcds.utils import between_sampler

_attrs = dict(id='id', source='source', target='target', key='key')

entity_classes = []
relationship_classes = []
schema = RSchema(entity_classes, relationship_classes)

# def json_for_schema(schema: RSchema, file=None):
#     ug = schema.as_networkx_ug(True)
#     ic_mapping = dict(zip(sorted(schema.item_classes), count()))
#     i_mapping = dict(zip(ug, count()))
#
#     data = {}
#
#     data['nodes'] = [dict(chain(ug.node[v].items(), [('name', str(v)),
#                                                      ('item_class', ic_mapping[v.item_class]),
#                                                      ('type', isinstance(v.item_class, E_Class))]))
#                      for v in ug]
#
#     data['links'] = [dict(chain(d.items(), [('source', i_mapping[u]),
#                                             ('target', i_mapping[v])]))
#                      for u, v, d in ug.edges_iter(data=True)]
#
#     if file is not None:
#         print(json.dumps(data), file=file)
#     else:
#         return data


def json_for_skeleton(skeleton: RSkeleton, file=None, no_fills=frozenset()):
    ug = skeleton.as_networkx_ug()
    ic_mapping = dict(zip(sorted(skeleton.schema.item_classes), count()))
    i_mapping = dict(zip(ug, count()))

    data = {}

    data['nodes'] = [dict(chain(ug.node[v].items(), [('name', str(v)),
                                                     ('item_class', ic_mapping[v.item_class]),
                                                     ('is_entity', isinstance(v.item_class, E_Class)),
                                                     ('no_fill', v in no_fills)]))
                     for v in ug]

    data['links'] = [dict(chain(d.items(), [('source', i_mapping[u]),
                                            ('target', i_mapping[v])]))
                     for u, v, d in ug.edges_iter(data=True)]

    if file is not None:
        print(json.dumps(data), file=file)
    else:
        return data


def json_for_ground_graph(ground_graph: GroundGraph, file=None):
    dag = ground_graph.as_networkx_dag()
    ic_mapping = dict(zip(sorted(ground_graph.schema.item_classes), count()))
    i_mapping = dict(zip(dag, count()))

    data = {}

    data['nodes'] = [dict(chain(dag.node[v].items(), [('name', str(v)),
                                                      ('item_class', ic_mapping[v[0].item_class]),
                                                      ('is_entity', isinstance(v[0].item_class, E_Class))]))
                     for v in dag]

    data['links'] = [dict(chain(d.items(), [('source', i_mapping[u]),
                                            ('target', i_mapping[v])]))
                     for u, v, d in dag.edges_iter(data=True)]

    if file is not None:
        print(json.dumps(data), file=file)
    else:
        return data


if __name__ == "__main__":
    schema = generate_schema(num_attr_classes_per_ent_class_distr=between_sampler(1, 1))
    print(repr(schema))
    # with open("../../web/schema.json", "w") as f:
    #     json_for_schema(schema, f)

    skeleton = generate_skeleton(schema, 50)
    with open("../../web/skeleton.json", "w") as f:
        json_for_skeleton(skeleton, f)

    rcm = generate_rcm(schema, num_dependencies=len(schema.entities) + 1, max_hop=5)
    for d in rcm.directed_dependencies:
        print(d)
    gg = GroundGraph(rcm, skeleton)

    with open("../../web/gg.json", "w") as f:
        json_for_ground_graph(gg, f)


    def pairs():
        for d1 in rcm.directed_dependencies:
            for d2 in rcm.directed_dependencies:
                for PyVx in (d1, reversed(d1)):
                    for QzVy in (d2, reversed(d2)):
                        if PyVx.cause.attr == QzVy.effect.attr:
                            yield PyVx, QzVy


    counter = count()

    for PyVx, QzVy in pairs():
        for cut, JJ in canonical_unshielded_triples(rcm, PyVx, QzVy, single=False, with_anchors=True):
            jj_skeleton, auxs = anchors_to_skeleton(schema, PyVx.cause.rpath, QzVy.cause.rpath, JJ)
            with open("../../web/cuts/jj_skeleton_" + str(next(counter)) + ".json", "w") as f:
                json_for_skeleton(jj_skeleton, f, auxs)
