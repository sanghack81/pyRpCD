import itertools
from itertools import groupby

import networkx as nx
import numpy as np
from numpy import diag, zeros, ix_, median
from numpy.core.umath import sqrt
from numpy.linalg import eigh, inv, LinAlgError
from scipy.sparse import dok_matrix

from pygk.utils import repmat, as_column, Lookup
from pyrcds.domain import RSkeleton


# d is for a distance function
# k is for a kernel function
# K is for a Kernel matrix
# D is for a distance matrix

def triangle_fixing(D, epsilon=1.0e-6):
    """triangle fixing for l2-norm

    References
    ----------
    [1] Triangle Fixing Algorithms for the Metric Nearness Problem
        Inderjit S. Dhillon, Suvrit Sra, and Joel A. Tropp
        NIPS 2005
    """
    n = len(D)
    for i, j in itertools.combinations(range(n), 2):
        assert D[i, i] == 0
        assert D[i, j] == D[j, i]

    z = dict()
    e = zeros((n, n))
    for i, j, k in itertools.combinations(range(n), 3):
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


def robust_eigh(K, level=1.0e-12):
    if np.any(np.isnan(K)):
        raise ValueError('NaN in the matrix.')
    if not (-float('inf') < np.min(K) <= np.max(K) < float('inf')):
        raise ValueError('Positive or negative infinity is in the matrix.')

    try:
        return eigh(K)
    except LinAlgError as e:
        print('LinAlgError: {}'.format(e))
        print('robust eigh with {} level noise.'.format(level))
        noise = np.random.randn(*K.shape)
        return robust_eigh(K + level * noise, level * 2)


def min_eigen_value(K):
    w, _ = robust_eigh(K)
    return min(w)


def denoise(K):
    w, v = robust_eigh(K)
    w[w < 0] = 0
    return v @ diag(w) @ inv(v)


def shift(K):
    w, v = robust_eigh(K)
    min_w = np.min(w)
    if min_w < 0:
        w -= min_w
    return v @ diag(w) @ inv(v)


def diffusion(K):
    w, v = robust_eigh(K)
    w = np.exp(w)
    return v @ diag(w) @ inv(v)


def flip(K):
    w, v = robust_eigh(K)
    w = np.abs(w)
    return v @ diag(w) @ inv(v)


def geomean(*Ks):
    full = Ks[0]
    for K in Ks[1:]:
        full *= K
    return np.power(full, 1 / len(Ks))


#
# def kernelize(base_kfun):
#     return lambda x, y=None: pairwise_kernels(x, y, metric=base_kfun)
#
#
# def kernel_matrix(kernel_func, X, Y=None):
#     """create a kernel matrix (numpy ndarray)
#
#
#     :param kernel_func: a kernel function
#     :param X: array-like
#     :param Y: array-like
#     :return: a kernel matrix
#     """
#     if Y is None:
#         K = array([len(X), len(X)])
#         for i, x in enumerate(X):
#             for j, y in enumerate(X[i:], i):
#                 K[j, i] = K[i, j] = kernel_func(x, y)
#         return K
#     else:
#         K = array([len(X), len(Y)])
#         for i, x in enumerate(X):
#             for j, y in enumerate(Y):
#                 K[i, j] = kernel_func(x, y)
#         return K


# def decomposition(xs: list, max_size=None) -> list:
#     if max_size is None:
#         max_size = len(xs)
#     return [comb for i in range(1, max_size + 1) for comb in combinations(xs, i)]
#
#
# def identity_kernel_func(x, y):
#     return 1.0 if x == y else 0.0
#
#
# def r_convolution_kernel_func(xs, ys, kernel=identity_kernel_func):
#     minlen = min(len(xs), len(ys))
#     decomp_x = decomposition(xs, minlen)
#     decomp_y = decomposition(ys, minlen)
#     k = 0.0
#     for x in decomp_x:
#         for y in decomp_y:
#             k += kernel(x, y)
#     return k


def normalize_by_diag(k):
    if k is None:
        return None

    x = sqrt(repmat(as_column(diag(k)), 1, len(k)))
    k = (k / x) / x.transpose()
    return k


def hausdorff_distance(xs, ys, d=(lambda a, b: abs(a - b))):
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
        perms = list(itertools.permutations(xs))
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
        mates = nx.max_weight_matching(g)
        ijs = sorted(filter(lambda ij: ij[1] >= len(xs), mates.items()),
                     key=lambda ij: ij[1])
        order_of_x = [ij[0] for ij in ijs]
        return (np.array(xs)[order_of_x]).tolist()


def eq_size_max_matching_distance(xs, ys, d=(lambda a, b: abs(a - b)),
                                  similarity=(lambda a, b: np.exp(-(a - b) * (a - b)))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        xs = tuple(xs)
        ys = tuple(ys)
        return sum(d(x, y) for x, y in zip(max_match(xs, ys, similarity), ys))


def eq_size_max_matching_min_distance(xs, ys, d=(lambda a, b: abs(a - b)),
                                      similarity=(lambda a, b: np.exp(-(a - b) * (a - b)))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        xs = tuple(xs)
        ys = tuple(ys)
        return min(d(x, y) for x, y in zip(max_match(xs, ys, similarity), ys))


def eq_size_max_matching_max_distance(xs, ys, d=(lambda a, b: abs(a - b)),
                                      similarity=(lambda a, b: np.exp(-(a - b) * (a - b)))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        xs = tuple(xs)
        ys = tuple(ys)
        return max(d(x, y) for x, y in zip(max_match(xs, ys, similarity), ys))


def eq_size_hausdorff_distance(xs, ys, d=(lambda a, b: abs(a - b))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        d1 = max(min(d(x, y) for y in ys) for x in xs)
        d2 = max(min(d(x, y) for x in xs) for y in ys)
        return max(d1, d2)


def eq_size_min_perm_distance(xs, ys, d=(lambda a, b: abs(a - b))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) != len(ys):
        return float('inf')
    else:
        return min(
            sum(d(x, y) for x, y in zip(xs_perm, ys))
            for xs_perm in itertools.permutations(xs))


def max_min_perm_distance(xs, ys, d=(lambda a, b: abs(a - b))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) == 0 or len(ys) == 0:
        return float('inf')
    else:
        if len(xs) < len(ys):
            xs, ys = ys, xs
        # xs is shorter or equal
        return max(eq_size_min_perm_distance(xs_comb, ys, d) for xs_comb in itertools.combinations(xs, len(xs)))


def list_set_distances():
    return [hausdorff_distance,
            eq_size_hausdorff_distance,
            eq_size_max_matching_distance,
            eq_size_min_perm_distance,
            max_min_perm_distance,
            eq_size_max_matching_min_distance,
            eq_size_max_matching_max_distance
            ]


def set_distance_matrix(data, set_metric=hausdorff_distance, metric=(lambda x, y: abs(x - y))):  # k_set = sum sum skf
    n = len(data)
    D = zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            D[j, i] = D[i, j] = set_metric(data[i], data[j], metric)

    return D


def __group_indices_by(labels):
    gb = groupby(sorted(enumerate(labels), key=lambda x: x[1]), key=lambda x: x[1])
    return ((label, [pair[0] for pair in pairs]) for label, pairs in gb)


def weisfeiler_lehman_vertex_kernel(skeleton: RSkeleton, h=4):
    nodes = np.array(sorted(skeleton.items()))
    node_num = Lookup(nodes)

    N = len(nodes)

    # step 0
    # assignment
    lookup = Lookup()
    labels = np.array([lookup[v.item_class] for v in nodes])  # labels[i]
    # feature mapping
    K = np.zeros((N, N))
    for ll, idxs in __group_indices_by(labels):
        K[ix_(idxs, idxs)] = 1

    def long_label(ll, v, ne_v):
        return (ll[v], *sorted(ll[ne_v]))

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

    return normalize_by_diag(K), nodes


def median_except_diag(D, exclude_inf=True, default=1):
    return stat_except_diag(D, exclude_inf, default, median)


def mean_except_diag(D, exclude_inf=True, default=1):
    return stat_except_diag(D, exclude_inf, default, np.mean)


def stat_except_diag(D, exclude_inf=True, default=1, func=median):
    if D.ndim != 2:
        raise TypeError('not a matrix')
    if D.shape[0] != D.shape[1]:
        raise TypeError('not a square matrix')
    if len(D) <= 1:
        raise ValueError('No non-diagonal element')

    lower = D[np.tri(len(D), k=-1, dtype=bool)]
    upper = D.transpose()[np.tri(len(D), k=-1, dtype=bool)]
    non_diagonal = np.concatenate((lower, upper))
    if exclude_inf:
        non_diagonal = non_diagonal[non_diagonal != float('inf')]

    if len(non_diagonal):
        return func(non_diagonal)
    else:
        return default
