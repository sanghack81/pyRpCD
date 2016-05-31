# Relational Conditional Independence
# Relational, Kernel Conditional Independence Permutation Test
import math
import typing
from itertools import groupby

import numpy as np
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import pairwise_distances

from pygk.utils import as_column, repmat, Lookup
from pykcipt.kcipt import KCIPT, CIResult
from pyrcds.domain import RSkeleton, ImmutableRSkeleton
from pyrcds.model import RVar, flatten


# TODO when not to normalize kernel?


# def MMD(xs, ys, metric, filter_params=False):
#     # x in R^n
#     # y in R^n
#     Kx = pairwise_kernels(xs, metric=metric, filter_params=filter_params)
#     Ky = pairwise_kernels(ys, metric=metric, filter_params=filter_params)
#     Kxy = pairwise_kernels(xs, ys, metric=metric, filter_params=filter_params)
#
#     np.fill_diagonal(Kx, 0)
#     np.fill_diagonal(Ky, 0)
#     np.fill_diagonal(Kxy, 0)
#
#     return np.mean(Kx) + np.mean(Ky) - 2 * np.mean(Kxy)

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
# # TODO performance
# def r_convolution_kernel_func(xs, ys, kernel=identity_kernel_func):
#     minlen = min(len(xs), len(ys))
#     decomp_x = decomposition(xs, minlen)
#     decomp_y = decomposition(ys, minlen)
#     k = 0.0
#     for x in decomp_x:
#         for y in decomp_y:
#             k += kernel(x, y)
#     return k
#
#
# def kernel_matrix(kernel_func, X, Y=None):
#     """
#     create a kernel matrix (numpy ndarray)
#     :param kernel_func: a kernel function
#     :param X: array-like
#     :param Y: array-like
#     :return: a kernel matrix
#     """
#     if Y is None:
#         Y = X
#     matrix = np.array([len(X), len(Y)])
#     for i, x in enumerate(X):
#         for j, y in enumerate(Y):
#             matrix[i, j] = kernel_func(x, y)
#     return matrix
#
#
# def multiset_rbf_kernel(X, Y, gamma):
#     """
#     (numpy ndarray)
#     :param X:
#     :param Y:
#     :param gamma: gamma in exp(-gamma * || X-Y ||)
#     :return: (numpy ndarray)
#     """
#     matrix = np.array([len(X), len(Y)])
#
#     def rbf(x, y, gamma):
#         euc = np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)
#         euc *= -gamma
#         return math.exp(euc)
#
#     for i, x in enumerate(X):
#         for j, y in enumerate(Y):
#             matrix[i, j] = r_convolution_kernel_func(x, y, rbf)
#
#
# # X: array-like
# # Y: array-like
# def parallel_kernel_matrix(kernel_func, X, Y=None):
#     if Y is None:
#         Y = X
#     kernel_matrix_as_list = Parallel(n_jobs=8)(delayed(kernel_func)(x, y) for x in X for y in Y)
#     matrix = np.reshape(np.array(kernel_matrix_as_list), (len(X), len(Y)))
#     return matrix


def normalize(k):
    if k is None:
        return None

    x = np.sqrt(repmat(as_column(np.diag(k)), 1, len(k)))
    k = (k / x) / x.transpose()
    return k


def multiply(*args):
    temp = None
    for arg in args:
        if temp is None:
            temp = arg
        else:
            temp = temp * arg
    return temp


# # kS((a,b,c),      (d,e))           = k(a,d)+k(a,e)+k(b,d)+k(b,e)+k(c,d)+k(c,e)
# # kS((None,a,b,c), (None,d,e)) = k(a,d)+k(a,e)+k(b,d)+k(b,e)+k(c,d)+k(c,e)+k(None,None)
# def average_pairwise_kernel(xs, ys, k):
#     if len(xs) == len(ys) == 0:
#         return 1
#
#     if len(xs) == 0 or len(ys) == 0:
#         return 0
#
#     v = 0.  #
#     for x in xs:
#         for y in ys:
#             v += k(x, y)
#     return v / (len(xs) * len(ys))


def hausdorff_distance(xs, ys, d=(lambda a, b: abs(a - b))):
    if len(xs) == 0 and len(ys) == 0:
        return 0.0
    elif len(xs) == 0 or len(ys) == 0:
        return float('inf')
    else:
        d1 = max(min(d(x, y) for y in ys) for x in xs)
        d2 = max(min(d(x, y) for x in xs) for y in ys)
        return max(d1, d2)


# not a metric
# def sum_of_minimum_distance(xs, ys, d=(lambda a, b: abs(a - b))):
#     if len(xs) == 0 and len(ys) == 0:
#         return 0.0
#     elif len(xs) == 0 or len(ys) == 0:
#         return float('inf')
#     else:
#         d1 = sum(min(d(x, y) for y in ys) for x in xs)
#         d2 = sum(min(d(x, y) for x in xs) for y in ys)
#         return (d1 + d2) / (len(xs) + len(ys))


# RIBL distance is not a metric

def set_distance_matrix(data, set_metric=hausdorff_distance, metric=(lambda x, y: abs(x - y))):  # k_set = sum sum skf
    n = len(data)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            D[j, i] = D[i, j] = set_metric(data[i], data[j], metric)

    return D


# def weighted_set_kernel(VK, num, data, k):
#     # data [{(item, value)}]
#     # kG(item, item) = item topological similarity
#     n = len(data)
#     K = np.zeros((n, n))
#     for i, datum_i in enumerate(data):
#         for j, datum_j in data[i:]:
#             k_ij = 0
#             for item_a, value_a in datum_i:
#                 for item_b, value_b in datum_j:
#                     k_ij += VK[num[item_a], num[item_b]] * k(value_a, value_b)
#             K[j, i] = K[i, j] = k_ij
#
#     return normalize(K)


def emp_1d_rbf(attr, skeleton):
    dist = pairwise_distances(as_column(dump_values(attr, skeleton)))
    median_dist = np.median(dist)
    assert 0 < median_dist
    return lambda x, y: math.exp(-abs(x - y) / median_dist)


def dump_values(attr, skeleton):
    return np.array([item[attr] for item in skeleton.items(skeleton.schema.item_class_of(attr))])


# any smoothing?
def weisfeiler_lehman_vertex_kernel(skeleton: RSkeleton, h=4):
    nodes = list(skeleton.items())
    node_num = Lookup(nodes)

    N = len(nodes)

    # step 0
    # assignment
    lookup = Lookup()
    labels = [lookup[v.item_class] for v in nodes]  # labels[i]
    # feature mapping
    K = dok_matrix((N, N), dtype='float')
    for ll, idxs in group_indices_by(labels):
        K[np.ix_(idxs, idxs)] = 1

    def long_label(ll, v, ne_v):
        return (ll[v], *sorted(ll[ne_v]))

    def to_num(vs):
        return [node_num[v] for v in vs]

    # step 1 ~ h
    for step in range(1, h + 1):  # 1 <= step <= h
        # assignment
        lookup = Lookup()
        new_labels = [lookup[long_label(labels, i, sorted(to_num(skeleton.neighbors(v))))]
                      for i, v in enumerate(nodes)]

        for ll, idxs in group_indices_by(new_labels):
            K[np.ix_(idxs, idxs)] += 1

        # post-loop
        labels = new_labels

    return normalize(K), nodes


def group_indices_by(labels):
    gb = groupby(sorted(enumerate(labels), key=lambda x: x[1]), key=lambda x: x[1])
    return ((label, [pair[0] for pair in pairs]) for label, pairs in gb)


class CITester:
    def __init__(self):
        pass

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()) -> CIResult:
        raise NotImplementedError()

    @property
    def is_p_value_available(self):
        raise NotImplementedError()


class SetKernelRCITester(CITester):
    """Set kernel (multi-instance kernel) based tester"""

    def __init__(self, skeleton: RSkeleton, n_jobs=-1, maxed=None, **kwargs):
        self.skeleton = ImmutableRSkeleton(skeleton)
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.maxed = maxed

        self.median_dist = dict()
        for attr in skeleton.schema.attrs:
            dist = pairwise_distances(as_column(dump_values(attr, skeleton)))
            self.median_dist[attr] = np.median(dist)

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()) -> CIResult:
        assert x != y
        assert x not in zs and y not in zs

        data = flatten(self.skeleton, (x, y, *zs), with_base_items=False, value_only=True)
        if self.maxed is not None and len(data) > self.maxed:
            data = data[np.random.choice(len(data), self.maxed, replace=False), :]

        K = [None] * (2 + len(zs))
        for i, rvar in enumerate((x, y, *zs)):
            D = set_distance_matrix(data[:, i])
            K[i] = np.exp(-D / self.median_dist[rvar.attr])

        return KCIPT(K[0], K[1], multiply(*K[2:]), n_jobs=self.n_jobs, **self.kwargs)

    @property
    def is_p_value_available(self):
        return True

# class GraphKernelRCITester(CITester):
#     def __init__(self, skeleton: RSkeleton, attr_kernels: dict, h=4, alpha=0.05):
#         self.skeleton = ImmutableRSkeleton(skeleton)
#         self.VK, self.ordered_items = weisfeiler_lehman_vertex_kernel(self.skeleton, h, last_only=True)
#         self.number_of = {v: i for i, v in enumerate(self.ordered_items)}
#         self.attr_kernels = attr_kernels
#         self.alpha = alpha
#
#     def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()):
#         assert x != y
#         assert x not in zs and y not in zs
#         assert y.is_canonical or x.is_canonical
#         if x.is_canonical:
#             x, y = y, x
#
#         data = flatten(self.skeleton, (x, y, *zs), with_base_items=False, value_only=False)
#
#         K = [None] * (2 + len(zs))
#         for i, rvar in enumerate((x, y, *zs)):
#             K[i] = weighted_set_kernel(self.VK, self.number_of, data[:, i], self.attr_kernels[rvar.attr])
#
#         return KCIPT(K[0], K[1], normalize(multiply(*K[2:])), alpha=self.alpha)
#
#     @property
#     def is_p_value_available(self):
#         return True
