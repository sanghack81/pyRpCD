from pyrcds.model import RDep, PRCM, llrsp, RVar, eqint


def joinable(p, *args):
    for q in args:
        p = p ** q
        if p is None:
            return False
    return True


# written for readability
def canonical_unshielded_triples(PyVx: RDep, QzVy: RDep, M: PRCM, single=True):
    """Returns a CUT or generate CUTs

    :param PyVx:
    :param QzVy:
    :param M:
    :param single: whether to return a single CUT (default True)
    :return:
    """
    LL = llrsp

    Py, Vx = PyVx
    Qz, Vy = QzVy
    P, Y = Py
    Q, Z = Qz
    V, Y2 = Vy

    if Y != Y2:
        raise AssertionError("{} and {} do not share the common attribute class.".format(PyVx, QzVy))

    m, n = len(P), len(Q)
    l = LL(reversed(P), Q)
    a_x, b_x = m - l, l - 1

    # A set of candidate anchors
    J = set()
    for a in range(a_x + 1):  # 0 <= <= a_x
        for b in range(b_x, n):  # b_x <=  <= |Q|
            if P[a] == Q[b]:
                J.add((a, b))

    # the first characteristic anchor (a_r,b_r)
    for a_r, b_r in J:
        if not (LL(P[:a_r:-1], Q[b_r:]) == LL(P[a_r:], Q[b_r:]) == 1):
            continue
        if not joinable(P[:a_r], Q[b_r:]):
            continue
        RrZ = RVar(P[:a_r] ** Q[b_r:], Z)
        if RrZ in M.adj(Vx):
            continue

        l_alpha = LL(Q[b_x:b_r:-1], P[:a_r:-1])
        if l_alpha == 1:
            if eqint(P[a_r:a_x], Q[b_x:b_r:-1]):
                if single:
                    return Vx, {Py, RVar(P[:a_r] ** Q[:b_r:-1], Y)}, RrZ
                else:
                    yield Vx, {Py, RVar(P[:a_r] ** Q[:b_r:-1], Y)}, RrZ

        elif l_alpha < b_r - b_x + 1 and a_r < a_x and b_x < b_r:
            a_y, b_y = a_r - l_alpha + 1, b_r - l_alpha + 1

            # the second characteristic anchor
            for a_s, b_s in J:
                if not (a_s <= a_y and b_x < b_s <= b_y):
                    continue
                if not joinable(P[:a_s], Q[b_s:]):
                    continue
                RsZ = RVar(P[:a_s] ** Q[b_s:], Z)
                if RsZ in M.adj(Vx):
                    continue

                PA, PB, QA, QB = P[:a_s:-1], P[a_s:a_y], Q[b_s:b_y], Q[b_x:b_s:-1]

                if LL(PA, QA) > 1 or LL(PA, QB) > 1:
                    continue

                l_beta = LL(PB, QB)
                if (not eqint(PB, QA)) or l_beta == min(len(PB), len(QB)):
                    continue

                a_z, b_z = a_s + l_beta - 1, b_s - l_beta + 1
                # the third characteristic anchor
                for a_t, b_t in J:
                    if not (a_r < a_t <= a_x and b_x <= b_t < b_z):
                        continue
                    if not joinable(P[:a_s], Q[b_t:b_s:-1], P[a_r:a_t:-1], Q[b_r:]):
                        continue
                    RtZ = RVar(P[:a_s] ** Q[b_t:b_s:-1] ** P[a_r:a_t:-1] ** Q[b_r:], Z)
                    if RtZ in M.adj(Vx):
                        continue

                    PC, PD, QC, QD = P[a_r:a_t:-1], P[a_t:a_x], Q[b_t:b_z], Q[b_x:b_t:-1]

                    if LL(PC, QC) > 1 or LL(PD, QC) > 1:
                        continue

                    l_gamma = LL(PC, QD)
                    assert 1 <= l_gamma
                    if l_gamma == 1 and eqint(PD, QD) or 1 < l_gamma < min(len(PC),
                                                                           len(QD)) and a_t < a_x and b_x < b_t:
                        a_w, b_w = a_t - l_gamma + 1, b_t - l_gamma + 1

                        PP = {P,
                              P[:a_w] ** Q[:b_w:-1],
                              P[:a_s] ** Q[:b_s:-1],
                              P[:a_s] ** Q[b_t:b_s:-1] ** P[a_t:],
                              P[:a_s] ** Q[b_s:b_r] ** P[a_r:],
                              P[:a_s] ** Q[b_s:b_r] ** P[a_r:a_w] ** Q[:b_w:-1]}

                        PP_Y = {RVar(PP_i, Y) for PP_i in PP}
                        if single:
                            return Vx, PP_Y, RrZ
                        else:
                            yield Vx, PP_Y, RrZ
                            yield Vx, PP_Y, RsZ
                            yield Vx, PP_Y, RtZ


