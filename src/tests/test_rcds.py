import numpy as np

from pyrcds.domain import generate_schema
from pyrcds.model import generate_rcm
from pyrcds.rcds import canonical_unshielded_triples

if __name__ == '__main__':
    for i in range(20):
        schema = generate_schema()
        rcm = generate_rcm(schema, np.random.randint(1, 100), np.random.randint(1, 20), np.random.randint(0, 20))
        print(i)

        for d1 in rcm.directed_dependencies:
            for d2 in rcm.directed_dependencies:
                for PyVx in (d1, reversed(d1)):
                    for QzVy in (d2, reversed(d2)):
                        if PyVx.cause.attr == QzVy.effect.attr:
                            for CUT in canonical_unshielded_triples(rcm, PyVx, QzVy, single=False, with_anchors=False):
                                pass
                            for CUT, JJ in canonical_unshielded_triples(rcm, PyVx, QzVy, single=False,
                                                                        with_anchors=True):
                                print(CUT)
                                print(JJ)
                            for CUT in canonical_unshielded_triples(rcm, PyVx, QzVy, single=True, with_anchors=False):
                                pass
                            for CUT, JJ in canonical_unshielded_triples(rcm, PyVx, QzVy, single=True,
                                                                        with_anchors=True):
                                pass
