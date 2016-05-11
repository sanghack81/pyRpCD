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


# kS((a,b,c), (d,e)) = k(a,d)+k(a,e)+k(b,d)+k(b,e)+k(c,d)+k(c,e)+k(None,None)
def inner_set_kernel(xs, ys, k):
    v = 1.  #
    for x in xs:
        for y in ys:
            v += k(x, y)
    return v / (1 + len(xs) * len(ys))


def set_kernel(data, skf):  # k_set = sum sum skf
    n = len(data)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[j, i] = K[i, j] = inner_set_kernel(data[i], data[j], skf)

    return normalize(K)


def weighted_set_kernel(VK, num, data, k):
    # data [{(item, value)}]
    # kG(item, item) = item topological similarity
    n = len(data)
    K = np.zeros((n, n))
    for i, datum_i in enumerate(data):
        for j, datum_j in data[i:]:
            k_ij = 0
            for item_a, value_a in datum_i:
                for item_b, value_b in datum_j:
                    k_ij += VK[num[item_a], num[item_b]] * k(value_a, value_b)
            K[j, i] = K[i, j] = k_ij

    return normalize(K)


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
    for k, g in group_by(labels):
        K[np.ix_(g, g)] = 1

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

        for k, g in group_by(new_labels):
            K[np.ix_(g, g)] += 1

        # post-loop
        labels = new_labels

    return normalize(K), nodes


def group_by(new_labels):
    gb = groupby(sorted(enumerate(new_labels), key=lambda x: x[1]), key=lambda x: x[1])
    return [(k, list(g)) for k, g in gb]


class CITester:
    def __init__(self):
        pass

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()) -> CIResult:
        raise NotImplementedError()

    @property
    def is_p_value_available(self):
        raise NotImplementedError()


# TODO AbstractKernelRCITester

class SetKernelRCITester(CITester):
    """Set kernel (multi-instance kernel) based tester"""

    def __init__(self, skeleton: RSkeleton, attr_kernels=None, alpha=0.05, n_job=1):
        assert 0.0 <= alpha <= 1.0

        self.skeleton = ImmutableRSkeleton(skeleton)
        self.alpha = alpha
        self.n_job = n_job

        if attr_kernels is None:
            self.attr_kernels = {attr: emp_1d_rbf(attr, skeleton) for attr in skeleton.schema.attrs}
        else:
            assert all(attr_kernels[attr] is not None for attr in skeleton.schema.attrs)
            self.attr_kernels = attr_kernels

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()) -> CIResult:
        assert x != y
        assert x not in zs and y not in zs

        data = flatten(self.skeleton, (x, y, *zs), with_base_items=False, value_only=True)
        attr_kernels = [self.attr_kernels[rvar.attr] for rvar in (x, y, *zs)]

        K = [None] * (2 + len(zs))
        for i in range(data.shape[1]):
            K[i] = set_kernel(data[:, i], attr_kernels[i])

        return KCIPT(K[0], K[1], normalize(multiply(*K[2:])), alpha=self.alpha, n_job=self.n_job)

    @property
    def is_p_value_available(self):
        return True


class GraphKernelRCITester(CITester):
    def __init__(self, skeleton: RSkeleton, attr_kernels: dict, h=4, alpha=0.05):
        self.skeleton = ImmutableRSkeleton(skeleton)
        self.VK, self.ordered_items = weisfeiler_lehman_vertex_kernel(self.skeleton, h, last_only=True)
        self.number_of = {v: i for i, v in enumerate(self.ordered_items)}
        self.attr_kernels = attr_kernels
        self.alpha = alpha

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()):
        assert x != y
        assert x not in zs and y not in zs
        assert y.is_canonical or x.is_canonical
        if x.is_canonical:
            x, y = y, x

        data = flatten(self.skeleton, (x, y, *zs), with_base_items=False, value_only=False)

        K = [None] * (2 + len(zs))
        for i, rvar in enumerate((x, y, *zs)):
            K[i] = weighted_set_kernel(self.VK, self.number_of, data[:, i], self.attr_kernels[rvar.attr])

        return KCIPT(K[0], K[1], normalize(multiply(*K[2:])), alpha=self.alpha)

    @property
    def is_p_value_available(self):
        return True
