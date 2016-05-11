import matplotlib.pyplot as plt
import networkx as nx
from palettable import colorbrewer
from palettable.colorbrewer import BrewerMap

from pyrcds.domain import RSchema, RSkeleton, R_Class


def draw_schema(schema: RSchema):
    g = schema.as_networkx_ug()
    n = len(schema.item_classes)
    pos = nx.spring_layout(g)

    bm = colorbrewer.get_map('Set1', 'qualitative', min(max(3, n), 12))
    assert isinstance(bm, BrewerMap)

    for i, item_class in enumerate(sorted(schema.item_classes)):
        nx.draw_networkx_nodes(g, pos, nodelist=[item_class, ],
                               node_color=bm.hex_colors[i % 12])

    nx.draw_networkx_edges(g, pos)
    plt.show()


def draw_skeleton(skeleton: RSkeleton):
    g = skeleton.as_networkx_ug()
    n = len(skeleton.schema.item_classes)
    pos = nx.spring_layout(g)

    bm = colorbrewer.get_map('Set1', 'qualitative', min(max(3, n), 12))
    assert isinstance(bm, BrewerMap)

    for i, item_class in enumerate(sorted(skeleton.schema.item_classes)):
        if isinstance(item_class, R_Class):
            nx.draw_networkx_nodes(g, pos, nodelist=list(skeleton.items(item_class)),
                                   node_color=bm.hex_colors[i % 12], node_size=50)
        else:
            nx.draw_networkx_nodes(g, pos, nodelist=list(skeleton.items(item_class)),
                                   node_color=bm.hex_colors[i % 12])

    nx.draw_networkx_edges(g, pos)
    plt.show()
