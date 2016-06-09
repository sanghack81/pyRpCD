from itertools import groupby, combinations, permutations

import networkx as nx
import numpy as np
from numpy import diag, zeros, ix_
from numpy.core.umath import sqrt
from numpy.linalg import inv, eigh
from scipy.sparse import dok_matrix

from pygk.utils import repmat, as_column, Lookup
from pyrcds.domain import RSkeleton


# d is for a distance function
# k is for a kernel function
# K is for a Kernel matrix
# D is for a distance matrix


def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def triangle_fixing(D, epsilon=1.0e-6):
    """triangle fixing for l2-norm

    References
    ----------
    [1] Triangle Fixing Algorithms for the Metric Nearness Problem
        Inderjit S. Dhillon, Suvrit Sra, and Joel A. Tropp
        NIPS 2005
    """
    n = len(D)
    for i, j in combinations(range(n), 2):
        assert D[i, i] == 0
        assert D[i, j] == D[j, i]

    z = dict()
    e = zeros((n, n))
    for i, j, k in combinations(range(n), 3):
        z[(i, j, k)] = 0
        z[(j, k, i)] = 0
        z[(k, i, j)] = 0

    delta = 1 + epsilon
    while delta > epsilon:
        delta = 0
        for a in range(n):
            for b in range(a + 1, n):
                if D[a, b] == float('inf'):
                    continue
                for c in range(b + 1, n):
                    if D[b, c] == float('inf'):
                        continue
                    for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
                        b_ = D[k, i] + D[j, k] - D[i, j]
                        mu = (e[i, j] - e[j, k] - e[k, i] - b_) / 3
                        theta = min(-mu, z[(i, j, k)])
                        e[i, j] += theta  # errata in [1]??
                        e[j, k] -= theta  # errata in [1]??
                        e[k, i] -= theta  # errata in [1]??
                        z[(i, j, k)] -= theta

                        e[j, i] = e[i, j]
                        e[k, j] = e[j, k]
                        e[i, k] = e[k, i]

                        delta += abs(theta)
    return D + e


def list_psd_converters():
    return [denoise, shift, diffusion, flip]


def min_eigen_value(K):
    w, _ = eigh(K)
    return min(w)


def denoise(K):
    w, v = eigh(K)
    w[w < 0] = 0
    return v @ diag(w) @ inv(v)


def shift(K):
    w, v = eigh(K)
    min_w = np.min(w)
    if min_w < 0:
        w -= min_w
    return v @ diag(w) @ inv(v)


def diffusion(K):
    w, v = eigh(K)
    w = np.exp(w)
    return v @ diag(w) @ inv(v)


def flip(K):
    w, v = eigh(K)
    w = np.abs(w)
    return v @ diag(w) @ inv(v)


def geomean(*Ks):
    if len(Ks) == 0:
        raise ValueError('matrices are not given.')

    full = 1
    for K in Ks:
        full *= K
    return np.power(full, 1 / len(Ks))


def normalize_by_diag(k):
    if k is None:
        return None

    x = sqrt(repmat(as_column(diag(k)), 1, len(k)))
    k = (k / x) / x.transpose()
    return k


def hausdorff_distance(xs, ys, d=euclidean):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) == 0 or len(ys) == 0:
        return float('inf')
    else:
        d1 = max(min(d(x, y) for y in ys) for x in xs)
        d2 = max(min(d(x, y) for x in xs) for y in ys)
        return max(d1, d2)


def max_match(xs, ys, similarity):
    assert len(xs) == len(ys)
    assert not isinstance(xs, set)
    assert not isinstance(ys, set)
    assert not isinstance(xs, frozenset)
    assert not isinstance(ys, frozenset)

    if len(xs) == 1:
        return xs
    elif len(xs) == 2:
        a, b = xs
        c, d = ys
        if similarity(a, c) + similarity(b, d) < similarity(b, c) + similarity(a, d):
            return (b, a)
        else:
            return (a, b)
    elif len(xs) == 3:
        a, b, c = xs
        d, e, f = ys
        max_perm = None
        max_sim = -float('inf')
        for x, y, z in (a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a):
            sim = similarity(x, d) + similarity(y, e) + similarity(z, f)
            if max_sim < sim:
                max_perm, max_sim = (x, y, z), sim
        return max_perm
    elif len(xs) <= 6:
        perms = list(permutations(xs))
        max_at = np.argmax([sum(similarity(x, y) for x, y in zip(p, ys))
                            for p in perms])
        return perms[max_at]
    else:
        # use maximum
        g = nx.Graph()
        g.add_nodes_from(range(len(xs) + len(ys)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys, start=len(xs)):
                g.add_edge(i, j, weight=similarity(x, y))
        mates = nx.max_weight_matching(g, True)
        ijs = sorted(filter(lambda ij: ij[1] >= len(xs), mates.items()),
                     key=lambda ij: ij[1])
        order_of_x = [ij[0] for ij in ijs]
        return (np.array(xs)[order_of_x]).tolist()


def eq_size_max_matching_distance(xs, ys, d=euclidean,
                                  similarity=(lambda a, b: np.exp(-(a - b) * (a - b)))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        xs = tuple(xs)
        ys = tuple(ys)
        return sum(d(x, y) for x, y in zip(max_match(xs, ys, similarity), ys))


def eq_size_max_matching_min_distance(xs, ys, d=euclidean,
                                      similarity=(lambda a, b: np.exp(-(a - b) * (a - b)))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        xs = tuple(xs)
        ys = tuple(ys)
        return min(d(x, y) for x, y in zip(max_match(xs, ys, similarity), ys))


def eq_size_max_matching_max_distance(xs, ys, d=euclidean,
                                      similarity=(lambda a, b: np.exp(-(a - b) * (a - b)))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        xs = tuple(xs)
        ys = tuple(ys)
        return max(d(x, y) for x, y in zip(max_match(xs, ys, similarity), ys))


def eq_size_hausdorff_distance(xs, ys, d=euclidean):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        d1 = max(min(d(x, y) for y in ys) for x in xs)
        d2 = max(min(d(x, y) for x in xs) for y in ys)
        return max(d1, d2)


def eq_size_min_perm_distance(xs, ys, d=euclidean):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        return min(
            sum(d(x, y) for x, y in zip(xs_perm, ys))
            for xs_perm in permutations(xs))


def max_min_perm_distance(xs, ys, d=euclidean):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) == 0 or len(ys) == 0:
        return float('inf')
    else:
        if len(xs) < len(ys):
            xs, ys = ys, xs
        # xs is shorter or equal
        return max(eq_size_min_perm_distance(xs_comb, ys, d) for xs_comb in combinations(xs, len(xs)))


def list_set_distances():
    return [hausdorff_distance,
            eq_size_hausdorff_distance,
            eq_size_max_matching_distance,
            eq_size_min_perm_distance,
            max_min_perm_distance,
            eq_size_max_matching_min_distance,
            eq_size_max_matching_max_distance
            ]


def set_distance_matrix(data, set_metric=hausdorff_distance,
                        element_metric=euclidean):  # k_set = sum sum skf
    n = len(data)
    D = zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            D[j, i] = D[i, j] = set_metric(data[i], data[j], element_metric)
    return D


def distance_matrix(data, metric=euclidean):
    n = len(data)
    D = zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            D[j, i] = D[i, j] = metric(data[i], data[j])
    return D


def __group_indices_by(labels):
    gb = groupby(sorted(enumerate(labels), key=lambda x: x[1]), key=lambda x: x[1])
    return ((label, [pair[0] for pair in pairs]) for label, pairs in gb)


# def split_vertex_kernel(skeleton: RSkeleton, vertex_kernel, **options):
#     all_items = np.array(sorted(skeleton.items()))
#     items_by_class = list(group_by(all_items, lambda skitem: skitem.item_class))
#     Ks = {item_class: vertex_kernel(skeleton, items, **options)
#           for item_class, items in items_by_class}
#
#     K = block_diag(Ks)
#     ordered_nodes = chain(*[items for _, items in items_by_class])
#     return K, ordered_nodes


# def dummy_vertex_kernel(skeleton, items, **unused_options):
#     return np.ones(len(items), len(items))


def weisfeiler_lehman_vertex_kernel(skeleton: RSkeleton, h=4, **unused_options):
    """Weisfeiler-Lehman kernel for vertices up to given hops"""
    nodes = np.array(sorted(skeleton.items()))
    node_num = Lookup(nodes)

    N = len(nodes)

    # step 0, assignment
    lookup = Lookup()
    labels = np.array([lookup[v.item_class] for v in nodes])  # labels[i]

    K = dok_matrix((N, N))
    for ll, idxs in __group_indices_by(labels):
        K[ix_(idxs, idxs)] = 1

    def long_label(ll, v, ne_v):
        neighbor_labels = sorted(ll[ne_v])
        return (ll[v], *neighbor_labels)

    def to_num(vs):
        return [node_num[v] for v in vs]

    # step 1 ~ h
    for step in range(1, h + 1):  # 1 <= step <= h
        # assignment
        lookup = Lookup()
        new_labels = np.array([lookup[long_label(labels, i, sorted(to_num(skeleton.neighbors(v))))]
                               for i, v in enumerate(nodes)])

        for ll, idxs in __group_indices_by(new_labels):
            K[ix_(idxs, idxs)] += 1

        # post-loop
        labels = new_labels

    return K, nodes
