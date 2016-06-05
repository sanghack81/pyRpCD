import numpy as np

from pyrcds.domain import generate_schema, generate_skeleton, ImmutableRSkeleton
from pyrcds.model import generate_rcm, RPath, RVar, GroundGraph, terminal_set, RDep
from pyrcds.rcds import canonical_unshielded_triples, anchors_to_skeleton
from pyrcds.utils import group_by


def test_evidence_completeness2():
    print('.')
    schema = generate_schema()
    rcm = generate_rcm(schema, np.random.randint(1, 100), np.random.randint(1, 20), np.random.randint(0, 20))
    for PyVx in sorted(rcm.full_dependencies):
        for QzVy in sorted(rcm.full_dependencies):
            (P, Y), (_, X) = PyVx
            (Q, Z), (_, Y2) = QzVy
            if Y != Y2:
                continue
            for cut, J in sorted(canonical_unshielded_triples(rcm, PyVx, QzVy, False, True)):
                skeleton, _ = anchors_to_skeleton(schema, P, Q, J)
                gg = GroundGraph(rcm, skeleton)
                assert gg.unshielded_triples()


def test_evidence_completeness():
    schema = generate_schema()
    rcm = generate_rcm(schema, np.random.randint(1, 100), np.random.randint(1, 20), np.random.randint(0, 20))
    grouped = dict(group_by(rcm.full_dependencies, lambda d: d.attrfy()))
    skeleton = ImmutableRSkeleton(generate_skeleton(schema))
    gg = GroundGraph(rcm, skeleton)
    all_cuts = set(canonical_unshielded_triples(rcm, single=False))

    cut_by_xyz = dict(group_by(all_cuts, lambda cut: (cut[0].attr, next(iter(cut[1])).attr, cut[2].attr)))
    sorted1 = sorted(gg.unshielded_triples())
    print('total {} unshielded triples'.format(len(sorted1)))
    # only first 100
    for ut in sorted1:
        (i, X), (j, Y), (k, Z) = ut
        if (i, X) > (k, Z):
            i, X, k, Z = k, Z, i, X
        # i to j
        PP = set()
        for d in sorted(grouped[(Y, X)]):
            if j in terminal_set(skeleton, d, i):
                PP.add(d.cause.rpath)
        assert PP, 'check gg code or terminal set'
        QQ = set()
        for d in grouped[(Z, Y)]:
            if k in terminal_set(skeleton, d.cause, j):
                QQ.add(d.cause)
        assert QQ, 'check gg code or terminal set'
        # i to k
        if (Z, X) in grouped:
            for d in grouped[(Z, X)]:
                assert k not in terminal_set(skeleton, d, i), 'check gg code or terminal set'

        assert cut_by_xyz[(X, Y, Z)]
        for cut in cut_by_xyz[(X, Y, Z)]:
            Vx, PPy, Rz = cut
            if any(P in PP for P, _ in PPy):
                R, _ = Rz
                if Z != X:
                    assert RDep(Rz, Vx) not in rcm.directed_dependencies
                    assert reversed(RDep(Rz, Vx)) not in rcm.directed_dependencies
                # covered
                if k in terminal_set(skeleton, R, i):
                    break
        else:
            PyVx = RDep(RVar(next(iter(PP)), Y), RVar(RPath(schema.item_class_of(X)), X))
            QzVy = RDep(next(iter(QQ)), RVar(RPath(schema.item_class_of(Y)), Y))
            for cut, J in canonical_unshielded_triples(rcm, PyVx, QzVy, False, True):
                print(cut)
                print(J)
            print('no cut found for {}'.format(ut))
            assert False


def test_both(seed=None):
    if seed:
        np.random.seed(seed)
    test_evidence_completeness()
    test_evidence_completeness2()


if __name__ == '__main__':
    while True:
        test_both()
