import numpy as np

from pyrcds.domain import generate_schema
from pyrcds.model import generate_rcm
from pyrcds.rcds import canonical_unshielded_triples

if __name__ == '__main__':
    for i in range(1000):
        schema = generate_schema()
        rcm = generate_rcm(schema, np.random.randint(1, 100), np.random.randint(1, 20), np.random.randint(0, 20))
        print(i)

        for d1 in rcm.directed_dependencies:
            for d2 in rcm.directed_dependencies:
                for PyVx in (d1, reversed(d1)):
                    for QzVy in (d2, reversed(d2)):
                        if PyVx.cause.attr == QzVy.effect.attr:
                            _ = list(canonical_unshielded_triples(PyVx, QzVy, rcm, single=False))
