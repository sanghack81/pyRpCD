# Relational Conditional Independence
# Relational, Kernel Conditional Independence Permutation Test
import typing

from numpy import exp, array, zeros, sqrt
from numpy.random import choice
from sklearn.metrics.pairwise import pairwise_distances

from pygk.utils import as_column
from pykcipt.kcipt import KCIPT, KCIPTResult, KIPT
from pyrcds._spaces import set_distance_matrix, median_except_diag, triangle_fixing, denoise, \
    weisfeiler_lehman_vertex_kernel, eq_size_max_matching_distance
from pyrcds.domain import RSkeleton, ImmutableRSkeleton
from pyrcds.model import RVar, flatten


def multiply(*args):
    if len(args) == 0:
        return None
    temp = 1
    for arg in args:
        temp *= arg
    return temp


def dump_values(attr, skeleton):
    return array([item[attr] for item in skeleton.items(skeleton.schema.item_class_of(attr))])


class CITester:
    def __init__(self):
        pass

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()) -> KCIPTResult:
        raise NotImplementedError()

    @property
    def is_p_value_available(self):
        raise NotImplementedError()


class SetKernelRCITester(CITester):
    """Set kernel (multi-instance kernel) based tester"""

    def __init__(self, skeleton: RSkeleton, n_jobs=-1, maxed=None, **options):
        self.skeleton = ImmutableRSkeleton(skeleton)
        self.n_jobs = n_jobs
        self.options = options
        self.maxed = maxed

        self.median_dist = dict()
        for attr in skeleton.schema.attrs:
            dist = pairwise_distances(as_column(dump_values(attr, skeleton)))
            self.median_dist[attr] = median_except_diag(dist)

        self.use_denoise = False
        self.fix_triangle_inequality = False

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()) -> KCIPTResult:
        assert x != y
        assert x not in zs and y not in zs

        data = flatten(self.skeleton, (x, y, *zs), with_base_items=False, value_only=True)
        if self.maxed is not None and len(data) > self.maxed:
            data = data[choice(len(data), self.maxed, replace=False), :]

        K = [None] * (2 + len(zs))
        for i, rvar in enumerate((x, y, *zs)):
            D = set_distance_matrix(data[:, i])
            if self.fix_triangle_inequality:
                D = triangle_fixing(D, 1.0e-12)
            K[i] = exp(-(D * D) / median_except_diag(D))
            if self.use_denoise:
                K[i] = denoise(K[i])

        return KCIPT(K[0], K[1], multiply(*K[2:]), n_jobs=self.n_jobs, **self.options)

    @property
    def is_p_value_available(self):
        return True


def xxxx(Ds):
    X = zeros(next(iter(Ds)).shape)
    for D in Ds:
        X = X + D * D
    return sqrt(X)


class GraphKernelRCITester(CITester):
    def __init__(self, skeleton: RSkeleton, h=4, alpha=0.05, n_jobs=-1, **options):
        self.skeleton = ImmutableRSkeleton(skeleton)
        self.VK, self.ordered_items = weisfeiler_lehman_vertex_kernel(self.skeleton, h)
        self.number_of = {v: i for i, v in enumerate(self.ordered_items)}
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.options = options

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()):
        assert x != y
        assert x not in zs and y not in zs
        assert y.is_canonical or x.is_canonical
        if x.is_canonical:
            x, y = y, x

        item, value = 0, 1
        data = flatten(self.skeleton, (x, y, *zs), with_base_items=False, value_only=False, n_jobs=self.n_jobs)

        def similarity(a, b):
            return self.VK[self.number_of[a[item]], self.number_of[b[item]]]

        def squared_matching_distance(xs, ys, d):
            return eq_size_max_matching_distance(xs, ys, d, similarity=similarity)

        K = [None] * (2 + len(zs))
        Ds = [None] * (2 + len(zs))
        for i, rvar in enumerate((x, y, *zs)):
            D = set_distance_matrix(data[:, i], squared_matching_distance, lambda a, b: (a[value] - b[value]) ** 2)
            # if self.fix_triangle_inequality:
            #     D = triangle_fixing(D, 1.0e-12)
            divider = median_except_diag(D, default=1.)
            if divider == 0:
                divider = 1.0
            K[i] = exp(-(D * D) / divider)
            Ds[i] = D
            # if self.use_denoise:
            #     K[i] = denoise(K[i])

        # return KCIPT(K[0], K[1], multiply(*K[2:]), n_jobs=self.n_jobs, **self.kwargs)
        if zs:
            return KCIPT(K[0] * multiply(*K[2:]), K[1], xxxx(Ds[2:]), n_jobs=self.n_jobs, **self.options)
        else:
            return KIPT(K[0], K[1], n_jobs=self.n_jobs, **self.options)

    @property
    def is_p_value_available(self):
        return True
